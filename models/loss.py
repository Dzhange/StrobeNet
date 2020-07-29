import torch.nn as nn
import torch

class L2Loss(nn.Module):
    def __init__(self):
        super(L2Loss, self).__init__()

    def forward(self, pred, target, detach=True):
        """
        mask is 1 channel
        """
        mask = target[:, -1, :, :].clone().requires_grad_(True).unsqueeze(1)

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

    def forward(self, output, target):
        return self.computeLoss(output, target)

    def computeLoss(self, output, target):
        OutIm = output
        nChannels = OutIm.size(1)
        TargetIm = target
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
        batch_size = target.size(0)
        target_mask = target[:, -1, :, :]
        out_mask = output[:, -1, :, :].clone().requires_grad_(True)
        out_mask = self.Sigmoid(out_mask)

        mask_loss = self.MaskLoss(out_mask, target_mask)

        target_img = target[:, :-1, :, :].detach()
        out_img = output[:, :-1, :, :].clone().requires_grad_(True)

        diff = out_img - target_img
        diff_norm = torch.norm(diff, p=self.P, dim=1)  # Same size as WxH
        masked_diff_norm = torch.where(out_mask > self.Thresh, diff_norm,
                                        torch.zeros(diff_norm.size(), device=diff_norm.device))
        nocs_loss = 0
        for i in range(0, batch_size):
            num_non_zero = torch.nonzero(masked_diff_norm[i]).size(0)
            if num_non_zero > 0:
                nocs_loss += torch.sum(masked_diff_norm[i]) / num_non_zero
            else:
                nocs_loss += torch.mean(diff_norm[i])
        loss = (self.MaskWeight*mask_loss) + (self.ImWeight*(nocs_loss / batch_size))
        return loss

class L2MaskLoss(LPMaskLoss):
    def __init__(self, Thresh=0.7, MaskWeight=0.7, ImWeight=0.3): # PARAM
        super().__init__(Thresh, MaskWeight, ImWeight, P=2)