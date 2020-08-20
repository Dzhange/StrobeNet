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
        # print(output.shape)
        # print(target.shape)
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
            Range = range(4*i, 4*(i+1))
            out = OutIm[:, Range, :, :]
            tar = TargetIm[:, Range, :, :]
            TotalLoss += self.computeMaskedLPLoss(out, tar)
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

        nocs_loss *= 10
        # print(occ_loss, nocs_loss)
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

        batch_size = out_img.size(0)
        total_loss = 0
        den = 0
        num_out_imgs = int(num_channels / 4)
        for i in range(0, num_out_imgs):
            Range = list(range(4*i, 4*(i+1)))
            total_loss += self.computeMaskedLPLoss(out_img[:, Range, :, :], target_img[:, Range, :, :])
            den += 1

        total_loss /= float(den)

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

class LBSLoss(nn.Module):
    """
        calculate the loss for dense pose estimation
        output structure:
            [0:3]: nocs
            [3]: mask            
            [4:4+16*3]: joints position
            [4+16*3:4+16*6]: joint direction
            [4+16*6:4+16*7]: skinning weights
            [4+16*7:20+16*8]: confidence

        target structure:
            maps = target['map']
                maps[0:3]: nocs
                maps[3]: mask 
                maps[4:4+16*1]: skinning weights
                maps[4+16*1:4+16*4]: joints position
                maps[4+16*4:4+16*7]: joint direction
                
                            
            pose = target['pose']
                pose[0:3] location
                pose[3:6] rotation
    """

    def __init__(self, bone_num=16):
        super().__init__()
        self.mask_loss = nn.BCELoss(reduction='mean')
        self.sigmoid = nn.Sigmoid()
        self.Thresh = 0.7
        self.bone_num = bone_num

    def forward(self, output, target):

        n_batch = output.shape[0]
        bone_num = self.bone_num
        loss = torch.Tensor([0]).to(device=output.device)

        pred_nocs = output[:, 0:3, :, :].clone().requires_grad_(True)
        out_mask = output[:, 3, :, :].clone().requires_grad_(True)
        out_mask = self.sigmoid(out_mask)

        pred_joint_score = output[:, 4:4+bone_num*1, :, :].clone().requires_grad_(True)

        pred_loc_map = output[:, 4+bone_num*1:4+bone_num*4, :, :].clone().requires_grad_(True)
        pred_rot_map = output[:, 4+bone_num*4:4+bone_num*7, :, :].clone().requires_grad_(True)
        pred_skin_weights = output[:, 4+bone_num*7:4+bone_num*8, :, :].clone().requires_grad_(True)

        tar_maps = target['maps']
        tar_pose = target['pose']

        target_nocs = tar_maps[:, 0:3, :, :]
        target_mask = tar_maps[:, 3, :, :]

        tar_skin_weights = tar_maps[:, 4:4+bone_num*1, :, :]
        
        tar_loc_map = tar_maps[:, 4+bone_num*1:4+bone_num*4, :, :]
        tar_rot_map = tar_maps[:, 4+bone_num*4:4+bone_num*7, :, :]
        
        tar_loc = tar_pose[:, :, 0:3]
        tar_rot = tar_pose[:, :, 3:6]
        # print(tar_pose.shape)

        mask_loss = self.mask_loss(out_mask, target_mask)
        # loss = mask_loss * 0.7 * 0.5

        nocs_loss = self.masked_l2_loss(pred_nocs, target_nocs, target_mask)
        # loss += nocs_loss * 0.3 * 0.5

        skin_loss = self.masked_l2_loss(self.sigmoid(pred_skin_weights), tar_skin_weights, target_mask)
        # print(skin_loss)
        loss += skin_loss
        
        skin_sum = pred_skin_weights.sum(dim=1)
        skin_bound_loss = torch.max(torch.zeros(1).to(device=skin_sum.device), skin_sum-1).mean()
        # loss += skin_bound_loss * 0.5

        # print("begin loc mp")
        loc_map_loss = self.masked_l2_loss(pred_loc_map, tar_loc_map, target_mask)
        # loss += loc_map_loss
        # print("end loc mp")
        # print(loss)
        rot_map_loss = self.masked_l2_loss(pred_rot_map, tar_rot_map, target_mask)
        
        joint_loc_loss = self.pose_nocs_loss(pred_loc_map,
                                            pred_joint_score,
                                            target_mask,
                                            tar_loc)
        
        # err = self.check_map(tar_loc_map,                                 
        #                     target_mask,
        #                     tar_loc)
        # print(err)
        # loss += joint_loc_loss * 0.25

        joint_rot_loss = self.pose_nocs_loss(pred_rot_map,
                                            pred_joint_score,
                                            target_mask,
                                            tar_rot)


        # print(mask_loss, nocs_loss, skin_loss + loc_map_loss, rot_map_loss, joint_loc_loss, joint_rot_loss)
        # loss = mask_loss + nocs_loss\
        #         + skin_loss + loc_map_loss + rot_map_loss\
        #         + joint_loc_loss + joint_rot_loss


        return loss


    def masked_l2_loss(self, out, tar, mask):

        batch_size = out.shape[0]
        
        out = out.clone().requires_grad_(True)
        diff = out - tar
        diff_norm = torch.norm(diff, 2, dim=1)
        masked_diff_norm = torch.where(mask > self.Thresh, diff_norm,
                                        torch.zeros(diff_norm.size(), device=diff_norm.device))
        l2_loss = 0
        for i in range(0, batch_size):
            num_non_zero = torch.nonzero(masked_diff_norm[i]).size(0)
            if num_non_zero > 0:
                l2_loss += torch.sum(masked_diff_norm[i]) / num_non_zero
            else:
                l2_loss += torch.mean(diff_norm[i])

        return l2_loss


    def pose_nocs_loss(self, pred_joint_map, pred_joint_score, out_mask, tar_joints):
        
        n_batch = pred_joint_map.shape[0]
        bone_num = self.bone_num

        # get final prediction: score map summarize
        pred_joint_map = pred_joint_map.reshape(n_batch, bone_num, 3, pred_joint_map.shape[2],
                                                pred_joint_map.shape[3])  # B,bone_num,3,R,R

        pred_joint_map = pred_joint_map * out_mask.unsqueeze(1).unsqueeze(1)

        # print(self.sigmoid(pred_joint_score).shape)
        # print(out_mask.unsqueeze(1).shape)
        pred_joint_score = self.sigmoid(pred_joint_score) * out_mask.unsqueeze(1)

        pred_score_map = pred_joint_score / (torch.sum(pred_joint_score.reshape(n_batch, bone_num, -1),
                                                    dim=2, keepdim=True).unsqueeze(3) + 1e-5)

        pred_joint_map = pred_joint_map.detach() * pred_score_map.unsqueeze(2)
        pred_joint = pred_joint_map.reshape(n_batch, bone_num, 3, -1).sum(dim=3)  # B,22,3
        joint_diff = torch.sum((pred_joint - tar_joints) ** 2, dim=2)  # B,22
        
        joint_loc_loss = joint_diff.sum() / (n_batch * pred_joint_map.shape[1])

        return joint_loc_loss


    # def check_map(self, tar_joint_map, out_mask, tar_joints):
        
    #     n_batch = tar_joint_map.shape[0]
    #     bone_num = self.bone_num
        
    #     # for i in range(16):
    #         # for j in range(3):

    #             # print(tar_joint_map[0,i*3+j,0,0], tar_joints[0,i,j])                

    #     tar_joint_map = tar_joint_map.reshape(n_batch, bone_num, 3, tar_joint_map.shape[2],
    #                                             tar_joint_map.shape[3])  # B,bone_num,3,R,R

        
    #     tar_joint_map = tar_joint_map * out_mask.unsqueeze(1).unsqueeze(1)


    #     pred_joint = tar_joint_map.reshape(n_batch, bone_num, 3, -1).mean(dim=3)  # B,22,3

    #     # print(pred_joint.shape, tar_joints.shape)
    #     joint_diff = torch.sum((pred_joint - tar_joints) ** 2, dim=2)  # B,22
        
    #     joint_loc_loss = joint_diff.sum() / (n_batch * tar_joint_map.shape[1])

    #     return joint_loc_loss

class Discarded_LBSLoss(nn.Module):
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


    # def computePoseLoss()

    def DISCARDED_computePoseLoss(self, output, target, target_kine, mask):
        
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
