import torch.nn as nn
import torch

class L2Loss(nn.Module):
    def __init__(self):
        super(L2Loss, self).__init__()

    def forward(self, pred, target, detach=True):
        """
        mask is 1 channel
        """
        mask = pred[:, -1, :, :].clone().requires_grad_(True).unsqueeze(1)

        # assert mask.max() <= 1 + 1e-6

        if detach:
            target = target.detach()
        mask = mask.detach()

        assert pred.shape == target.shape
        dif = (pred - target) ** 2 * (mask.float())
        
        loss = torch.sum(dif.reshape(mask.shape[0], -1).contiguous(), 1)
        count = torch.sum(mask.reshape(mask.shape[0], -1).contiguous(), 1).detach()
        loss[count == 0] = loss[count == 0] * 0
        loss = loss / (count + 1)

        # this normalization is only for the image dimensions but not take channel dimension into account!
        non_zero_count = torch.sum((count > 0).float())
        if non_zero_count == 0:
            loss = torch.sum(loss) * 0
        else:
            loss = torch.sum(loss) / non_zero_count
        return loss


class LPMaskLoss(nn.Module):
    Thresh = 0.7 # PARAM
    def __init__(self, Thresh=0.7, MaskWeight=0.7, ImWeight=0.3, P=2): # PARAM
        super().__init__()
        self.P = P
        self.MaskLoss = nn.BCELoss(reduction='mean')
        self.Sigmoid = nn.Sigmoid()
        self.Thresh = Thresh
        self.MaskWeight = MaskWeight
        self.ImWeight = ImWeight
        self.DebugLoss = nn.MSELoss()

    def forward(self, output, target):
        # return self.computeLoss(output, target)
        return self.computeLossDebug(output, target)

    def computeLossDebug(self,output,target):
        # print(output)
        # print(target)
        return self.DebugLoss(output,target[0])

    def computeLoss(self, output, target):
        OutIm = output
        nChannels = OutIm.size(1)
        TargetIm = target[0]
        if nChannels%4 != 0:
            raise RuntimeError('Empty or mismatched batch (should be multiple of 4). Check input size {}.'.format(OutIm.size()))
        if nChannels != TargetIm.size(1):
            raise RuntimeError('Out target {} size mismatch with nChannels {}. Check input.'.format(TargetIm.size(1), nChannels))

        BatchSize = OutIm.size(0)
        TotalLoss = 0
        Den = 0
        nOutIms = int(nChannels / 4)
        for i in range(0, nOutIms):
            Range = list(range(4*i, 4*(i+1)))
            TotalLoss += self.computeMaskedLPLoss(OutIm[:, Range, :, :], TargetIm[:, Range, :, :])
            Den += 1

        TotalLoss /= float(Den)

        return TotalLoss

    def computeMaskedLPLoss(self, output, target):
        BatchSize = target.size(0)
        TargetMask = target[:, -1, :, :]
        OutMask = output[:, -1, :, :].clone().requires_grad_(True)
        OutMask = self.Sigmoid(OutMask)

        MaskLoss = self.MaskLoss(OutMask, TargetMask)

        TargetIm = target[:, :-1, :, :].detach()
        OutIm = output[:, :-1, :, :].clone().requires_grad_(True)

        Diff = OutIm - TargetIm
        DiffNorm = torch.norm(Diff, p=self.P, dim=1)  # Same size as WxH
        MaskedDiffNorm = torch.where(OutMask > self.Thresh, DiffNorm,
                                        torch.zeros(DiffNorm.size(), device=DiffNorm.device))
        NOCSLoss = 0
        for i in range(0, BatchSize):
            nNonZero = torch.nonzero(MaskedDiffNorm[i]).size(0)
            if nNonZero > 0:
                NOCSLoss += torch.sum(MaskedDiffNorm[i]) / nNonZero
            else:
                NOCSLoss += torch.mean(DiffNorm[i])

        Loss = (self.MaskWeight*MaskLoss) + (self.ImWeight*(NOCSLoss / BatchSize))
        return Loss

class L2MaskLoss(LPMaskLoss):
    def __init__(self, Thresh=0.7, MaskWeight=0.7, ImWeight=0.3): # PARAM
        super().__init__(Thresh, MaskWeight, ImWeight, P=2)        