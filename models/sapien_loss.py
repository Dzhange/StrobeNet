import os
import torch.nn.functional as F
import torch.nn as nn
import torch
from models.loss import *

class PMloss(LBSLoss):
    """
    Everything loss for parital mobility
    """
    def __init__(self, cfg, bone_num=2):
        super().__init__(cfg, bone_num)
        self.seg_loss = torch.nn.CrossEntropyLoss()

    def forward(self, output, target):
        bone_num = self.bone_num
        loss = torch.Tensor([0]).to(device=output.device)
        
        pred_skin_seg = output[:, 4+bone_num*6:4+bone_num*7, :, :].clone().requires_grad_(True)
        tar_maps = target['maps']
        target_mask = tar_maps[:, 3, :, :]
        tar_skin_seg = tar_maps[:, 4+bone_num*6:4+bone_num*6+1, :, :] ## as Seg
        # skin_loss = self.masked_l2_loss(self.sigmoid(pred_skin_weights), tar_skin_weights, target_mask)
        # skin_loss = self.seg_loss(pred_skin_seg, tar_skin_seg.squeeze(1).long().detach())
        input = pred_skin_seg.transpose(1, 2).transpose(2, 3).contiguous().view(-1, bone_num)
        skin_loss = self.seg_loss(input, tar_skin_seg.long().squeeze(1).view(-1))
        loss += skin_loss
        return loss


class PMLBSLoss(nn.Module):
    """
        calculate the loss for dense pose estimation
        output structure:
            [0:3]: nocs
            [3]: mask
            [4:4+bone_num*3]: joints position
            [4+bone_num*3:4+bone_num*6]: joint direction
            [4+bone_num*6:4+bone_num*7+2]: skinning weights
            [4+bone_num*7+2:4+bone_num*8+2]: confidence

        target structure:
            maps = target['map']
                maps[0:3]: nocs
                maps[3]: mask
                maps[4:4+bone_num*3]: joints position
                maps[4+bone_num*3:4+bone_num*6]: joint direction
                maps[4+bone_num*6:4+bone_num*7]: skinning weights

            pose = target['pose']
                pose[0:3] location
                pose[3:6] rotation
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.mask_loss = nn.BCELoss(reduction='mean')
        self.sigmoid = nn.Sigmoid()
        self.Thresh = 0.7
        self.bone_num = cfg.BONE_NUM
        self.l2_loss = JiahuiL2Loss()
        self.seg_loss = torch.nn.CrossEntropyLoss()

        self.expt_dir_path = os.path.join(self.cfg.OUTPUT_DIR, self.cfg.EXPT_NAME)
        if os.path.exists(self.expt_dir_path) == False:
            os.makedirs(self.expt_dir_path)

        # self.output_dir = os.path.join(self.expt_dir_path, "Check")
        # if os.path.exists(self.output_dir) == False:
        #     os.makedirs(self.output_dir)
        
        self.frame_id = 0

    def forward(self, output, target):
        
        loss = {}
        
        bone_num = self.bone_num
        # loss = torch.Tensor([0]).to(device=output.device)

        pred_nocs = output[:, 0:3, :, :].clone().requires_grad_(True)
        out_mask = output[:, 3, :, :].clone().requires_grad_(True)
        out_mask = self.sigmoid(out_mask)

        pred_loc_map = self.sigmoid(output[:, 4:4+bone_num*3, :, :].clone().requires_grad_(True))
        pred_rot_map = self.sigmoid(output[:, 4+bone_num*3:4+bone_num*6, :, :].clone().requires_grad_(True))

        pred_skin_seg = output[:, 4+bone_num*6:4+bone_num*7+2, :, :].clone().requires_grad_(True)
        pred_joint_score = output[:, 4+bone_num*7+2:4+bone_num*8+2, :, :].clone().requires_grad_(True)

        tar_maps = target['maps']
        target_nocs = tar_maps[:, 0:3, :, :] # nocs
        target_mask = tar_maps[:, 3, :, :] # mask
        tar_loc_map = self.sigmoid(tar_maps[:, 4:4+bone_num*3, :, :]) # location
        tar_rot_map = self.sigmoid(tar_maps[:, 4+bone_num*3:4+bone_num*6, :, :]) # rotation, angle axis
        tar_skin_seg = tar_maps[:, 4+bone_num*6:4+bone_num*6+1, :, :]

        tar_pose = target['pose']
        tar_loc = self.sigmoid(tar_pose[:, :, 0:3])
        tar_rot = self.sigmoid(tar_pose[:, :, 3:6])

        mask_loss = self.mask_loss(out_mask, target_mask)
        nocs_loss = self.masked_l2_loss(pred_nocs, target_nocs, target_mask)

        # skin_loss = self.masked_l2_loss(self.sigmoid(pred_skin_seg), tar_skin_weights, target_mask)
        pred_seg = pred_skin_seg.transpose(1, 2).transpose(2, 3).contiguous().view(-1, bone_num+2)
        tar_seg = tar_skin_seg.long().squeeze(1).view(-1)
        
        skin_loss = self.seg_loss(pred_seg, tar_seg)
        loc_map_loss = self.l2_loss(pred_loc_map, tar_loc_map, target_mask)
        rot_map_loss = self.l2_loss(pred_rot_map, tar_rot_map, target_mask)
                
        loc_loss = self.pose_nocs_loss(pred_loc_map,
                                            pred_joint_score,
                                            target_mask,
                                            tar_loc)
        rot_loss = self.pose_nocs_loss(pred_rot_map,
                                            pred_joint_score,
                                            target_mask,
                                            tar_rot)

        vis = 0
        if vis:
            self.vis_joint_loc(tar_loc_map, target_mask, self.frame_id)
            self.frame_id += 1
            if self.frame_id == 80:
                import sys
                sys.exit()
        
        loss['nox_loss'] = nocs_loss
        loss['mask_loss'] = mask_loss
        loss['loc_loss'] = loc_loss
        loss['loc_map_loss'] = loc_map_loss
        loss['rot_loss'] = rot_loss
        loss['rot_map_loss'] = rot_map_loss
        loss['skin_loss'] = skin_loss
        
        if output.shape[1] > 64 + 4+bone_num*8+2:
            pred_pnnocs = output[:, -3:, :, :].clone().requires_grad_(True)            
            tar_pnnocs = tar_maps[:, -3:, :, :]
            pnnocs_loss = self.masked_l2_loss(pred_pnnocs, tar_pnnocs, target_mask)            
            # loss += pnnocs_loss
            loss['pnnocs_loss'] = pnnocs_loss
        # print("[ DIFF ] map_loss is {:5f}; loc_loss is {:5f}".format(loc_map_loss, joint_loc_loss))

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
        pred_joint_score = self.sigmoid(pred_joint_score) * out_mask.unsqueeze(1)
        pred_score_map = pred_joint_score / (torch.sum(pred_joint_score.reshape(n_batch, bone_num, -1),
                                                    dim=2, keepdim=True).unsqueeze(3) + 1e-5)
                
        pred_joint_map = pred_joint_map.detach() * pred_score_map.unsqueeze(2)
        pred_joint = pred_joint_map.reshape(n_batch, bone_num, 3, -1).sum(dim=3)  # B,22,3
        
        joint_diff = torch.sum((pred_joint - tar_joints) ** 2, dim=2)  # B,22
        joint_loc_loss = joint_diff.sum() / (n_batch * pred_joint_map.shape[1])

        # joint_diff = pred_joint - tar_joints
        # diff_norm = torch.norm(joint_diff, p=2, dim=1)  # Same size as WxH
        # joint_loc_loss = torch.mean(diff_norm)

        return joint_loc_loss

    def vis_joint_map(self, joint_map, mask, frame_id):
        """
        save the inter results of joint predication as RGB image
        """
        bone_num = self.bone_num
        mask = mask.cpu().detach()

        zero_map = torch.zeros(3, joint_map.shape[2], joint_map.shape[3])

        to_cat = ()
        for i in range(bone_num):
            cur_bone = joint_map[0, i*3:i*3+3, :, :]
            gt = joint_map[0, i*3:i*3+3, :, :]

            joint_map = torch.where(mask > 0.7, cur_bone, zero_map)
            joint_map = torch2np(joint_map) * 255

            to_cat = to_cat + (joint_map, )

        big_img = np.concatenate(to_cat, axis=0)
        cv2.imwrite(os.path.join(self.output_dir, 'check_{}_pred_joint.png').format(str(frame_id).zfill(3)),
                        cv2.cvtColor(big_img, cv2.COLOR_BGR2RGB))

    def vis_joint_loc(self, joint_map, mask, frame_id):
        
        n_batch = joint_map.shape[0]
        bone_num = self.bone_num
        joint_map = joint_map.reshape(n_batch, bone_num, 3, joint_map.shape[2],
                                                joint_map.shape[3])  # B,bone_num,3,R,R
        joint_map = joint_map * mask.unsqueeze(1).unsqueeze(1)

        mean_gt_joint = joint_map.reshape(n_batch, bone_num, 3, -1).sum(dim=3)  # B,22,3
        mean_gt_joint /= mask.nonzero().shape[0]
        mean_gt_joint = mean_gt_joint[0]

        mean_gt_path = os.path.join(self.output_dir, 'check_{}_gt_mean_loc.xyz').format(str(frame_id).zfill(3))
        self.write(mean_gt_path, mean_gt_joint)

    def write(self, path, joint):
        f = open(path, "a")
        for i in range(joint.shape[0]):
            p = joint[i]
            f.write("{} {} {}\n".format(p[0], p[1], p[2]))
        f.close()

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

class PMLoss(nn.Module):
    """
    The Loss function for the whole lbs pipeline
    segnet loss would be used to supervise the first stage
    ifnet loss would supervise the final reconstrution error
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.segnet_loss = PMLBSLoss(config)

    def forward(self, output, target):
        segnet_output = output[0]
        loss = self.segnet_loss(segnet_output, target)

        ifnet_output = output[1]
        loss['recon_loss'] = self.recon_loss(ifnet_output, target)

        all_loss = self.add_up(loss)

        return all_loss

    def recon_loss(self, recon, target):

        occ = target['occupancies'].to(device=recon.device)
        # out = (B,num_points) by componentwise comparing vecots of size num_samples :)
        occ_loss = nn.functional.binary_cross_entropy_with_logits(
                recon, occ, reduction='none')
        num_sample = occ_loss.shape[1]
        occ_loss = occ_loss.sum(-1).mean() / num_sample

        return occ_loss
    
    
    def add_up(self, loss):
        
        all_loss = torch.zeros(1, device=loss['nox_loss'].device)
        # print(loss)
        cfg = self.config
        all_loss = all_loss\
            + cfg.NOCS_LOSS * (loss['nox_loss'] + loss['mask_loss'])\
            + cfg.LOC_LOSS * (loss['loc_loss'] + loss['loc_map_loss'])\
            + cfg.POSE_LOSS * (loss['rot_loss'] + loss['rot_map_loss'])\
            + cfg.SKIN_LOSS * loss['skin_loss']\

        if cfg.REPOSE == True:
            all_loss += loss['pnnocs_loss']
        if cfg.STAGE_ONE == False:
            all_loss += loss['recon_loss']

        # print("all loss is ", all_loss)
        # if all_loss > 3:
        #     print("error, outlier")
        return all_loss
    