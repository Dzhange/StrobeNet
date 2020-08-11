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

        # print("MaskLoss is {}".format(self.MaskWeight*mask_loss))
        # print("IMGLoss is {}".format(self.ImWeight*(nocs_loss / batch_size)))
        loss = (self.MaskWeight*mask_loss) + (self.ImWeight*(nocs_loss / batch_size))
        return loss

class L2MaskLoss(LPMaskLoss):
    def __init__(self, Thresh=0.7, MaskWeight=0.7, ImWeight=0.3): # PARAM
        super().__init__(Thresh, MaskWeight, ImWeight, P=2)

############################# Loss for vote network ###################################

class LBSLoss(nn.Module):
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
        out_img = output
        n_channels = out_img.size(1)
        target_img = target[0]
        pose = target[1]        
        # if n_channels != target_img.size(1):
            # raise RuntimeError('Out target {} size mismatch with nChannels {}. Check input.'.format(target_img.size(1), n_channels))        
        total_loss = 0
        
        batch_size = target_img.size(0)
        target_mask = target_img[:, 3, :, :]
        out_mask = output[:, 3, :, :].clone().requires_grad_(True) #take the 4th channel as mask
        out_mask = self.Sigmoid(out_mask)
        
        mask_loss = self.MaskLoss(out_mask, target_mask)
        nocs_loss = self.computeNOCSLoss(out_img, target_img, out_mask)
        # lbs_loss = self.computePoseLoss(out_img, target_img, pose, out_mask)

        # total_loss = mask_loss + nocs_loss + lbs_loss
        # total_loss = mask_loss + nocs_loss
        total_loss = (self.MaskWeight*mask_loss) + (self.ImWeight*(nocs_loss / batch_size))
        return total_loss

    def computeNOCSLoss(self, output, target, mask):

        batch_size = output.shape[0]
        target_img = target[:, :3, :, :].detach()
        out_img = output[:, :3, :, :].clone().requires_grad_(True)

        diff = out_img - target_img
        diff_norm = torch.norm(diff, p=self.P, dim=1)  # Same size as WxH
        masked_diff_norm = torch.where(mask > self.Thresh, diff_norm,
                                       torch.zeros(diff_norm.size(), device=diff_norm.device))
        nocs_loss = 0
        for i in range(0, batch_size):
            num_non_zero = torch.nonzero(masked_diff_norm[i]).size(0)
            if num_non_zero > 0:
                nocs_loss += torch.sum(masked_diff_norm[i]) / num_non_zero
            else:
                nocs_loss += torch.mean(diff_norm[i])
        return  nocs_loss 

    def computePoseLoss(self, output, target, target_kine, mask):
        
        # we have 16 bones
        # so starting from Channel 4(012 rgb, 3 mask)
        # channel 4:20 skining weights
        # channel 20:68 pose
        # channel 68:116 joint position
        bone_num = 16
        batch_size = output.shape[0]
        zero_map = torch.zeros(output.size(), device=output.device)        
        output = torch.where(mask.unsqueeze(1) > self.Thresh, output, zero_map)

        ######### skin part #################
        skin_weights = output[:, 4:20, :, :]
        target_skin_weights = target[:, 4:20, :, :]
        skin_diff_norm = torch.norm(skin_weights - target_skin_weights, p=self.P, dim=1)  # Same size as WxH
        skin_loss = 0
        for i in range(0, batch_size):
            num_non_zero = torch.nonzero(skin_diff_norm[i]).size(0)
            if num_non_zero > 0:
                skin_loss += torch.sum(skin_diff_norm[i]) / num_non_zero
            else:
                skin_loss += torch.mean(skin_diff_norm[i])

        ########## pose part ################# 
        pose_map = output[:, 20:68, :, :]
        target_pose = target_kine[:, :, :3]

        joint_map = output[:, 68:116, :, :]
        target_joint = target_kine[:, :, 3:6]
          
        # first vote for pose
        # pose is represented in axis_angle form
        pose_loss = 0
        joint_loss = 0
        for b in range(batch_size):
        
            for i in range(bone_num):
                pose_i_map = pose_map[b, 3*i:3*i+1, :, :]
                weighted_map = pose_i_map * skin_weights[b, i, :, :]
                num_non_zero = torch.nonzero(weighted_map).size(0)
                if num_non_zero > 0:
                    pred_pose = weighted_map.sum(dim=-1).sum(dim=-1) / num_non_zero
                else:
                    pred_pose = torch.zeros(3).to(device=output.device)
                pose_diff = target_pose[b, i] - pred_pose
                pose_loss += torch.mean(pose_diff.norm())
            
            for i in range(bone_num):
                joint_i_map = joint_map[b, 3*i:3*i+1, :, :]
                weighted_map = joint_i_map * skin_weights[b, i, :, :]
                num_non_zero = torch.nonzero(weighted_map).size(0)
                if num_non_zero > 0:
                    pred_joint = weighted_map.sum(dim=-1).sum(dim=-1) / num_non_zero
                else:
                    pred_joint = torch.zeros(3).to(device=output.device)
                joint_diff = target_joint[b, i] - pred_joint                
                joint_loss += torch.mean(joint_diff.norm())
        
        # lbs_loss = (skin_loss + pose_loss / bone_num + joint_loss / bone_num) / batch_size
        lbs_loss = (skin_loss) / batch_size
        return lbs_loss



class MixLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l2_mask_loss = L2MaskLoss()
        self.OccLoss = nn.functional.binary_cross_entropy_with_logits

    def forward(self, output, target):
        pred_nocs = output[0]

        nocs_loss = self.l2_mask_loss(pred_nocs, target['NOCS'])

        recon = output[1]
        # print(type(recon), recon.shape)
        occ = target['occupancies'].to(device=recon.device)

        occ_loss = self.OccLoss(recon, occ, reduction='none')# out = (B,num_points) by componentwise comparing vecots of size num_samples :)                

        num_sample = occ_loss.shape[1]
        occ_loss = occ_loss.sum(-1).mean() / num_sample

        return occ_loss + nocs_loss

class L2MaskLoss_wtFeature(nn.Module):
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
        out_img = output        
        num_channels = target.size(1) # use target size instead, we do not supervise feature channels
        target_img = target
        if num_channels%4 != 0:
            raise RuntimeError('Empty or mismatched batch (should be multiple of 4). Check input size {}.'.format(out_img.size()))

        BatchSize = out_img.size(0)
        total_loss = 0
        Den = 0
        num_out_imgs = int(num_channels / 4)
        for i in range(0, num_out_imgs):
            Range = list(range(4*i, 4*(i+1)))
            total_loss += self.computeMaskedLPLoss(out_img[:, Range, :, :], target_img[:, Range, :, :])
            Den += 1

        total_loss /= float(Den)

        return total_loss

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

        # print("MaskLoss is {}".format(self.MaskWeight*mask_loss))
        # print("IMGLoss is {}".format(self.ImWeight*(nocs_loss / batch_size)))
        loss = (self.MaskWeight*mask_loss) + (self.ImWeight*(nocs_loss / batch_size))
        return loss
