import os
import torch.nn.functional as F
import torch.nn as nn
import torch
from utils.DataUtils import *
from models.loss import *
from time import time

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
        # self.sigmoid = nn.Sigmoid()
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
        show_time = 0   

        bone_num = self.bone_num
        batch_size = output.shape[0]
        # loss = torch.Tensor([0]).to(device=output.device)

        pred_nocs = output[:, 0:3, :, :].clone().requires_grad_(True)
        out_mask = output[:, 3, :, :].clone().requires_grad_(True)
        out_mask = out_mask.sigmoid()

        pred_loc_map = output[:, 4:4+bone_num*3, :, :].clone().requires_grad_(True).sigmoid()
        pred_rot_map = output[:, 4+bone_num*3:4+bone_num*6, :, :].clone().requires_grad_(True).sigmoid()

        pred_skin_seg = output[:, 4+bone_num*6:4+bone_num*7+2, :, :].clone().requires_grad_(True)
        pred_joint_score = output[:, 4+bone_num*7+2:4+bone_num*8+2, :, :].clone().requires_grad_(True)

        tar_maps = target['maps']        
        target_nocs = tar_maps[:, 0:3, :, :] # nocs
        target_mask = tar_maps[:, 3, :, :] # mask
        tar_loc_map = tar_maps[:, 4:4+bone_num*3, :, :].sigmoid() # location
        tar_rot_map = tar_maps[:, 4+bone_num*3:4+bone_num*6, :, :].sigmoid() # rotation, angle axis
        tar_skin_seg = tar_maps[:, 4+bone_num*6:4+bone_num*6+1, :, :]
        # print(tar_maps.shape, tar_skin_seg.shape)
        # nocs_pc = pred_nocs[0, :, target_mask.squeeze() > 0].permute(1, 0)

        tar_pose = target['pose']
        tar_loc = tar_pose[:, :, 0:3].sigmoid()
        tar_rot = tar_pose[:, :, 3:6].sigmoid()
        
        if show_time:
            t1 = torch.cuda.Event(enable_timing=True)
            t2 = torch.cuda.Event(enable_timing=True)        
            t3 = torch.cuda.Event(enable_timing=True)        
        if show_time:
            torch.cuda.synchronize()
            t1.record()
        
            # mask_loss_2 = nn.functional.binary_cross_entropy(out_mask, target_mask)
                
            # torch.cuda.synchronize()
            # t2.record()

        mask_loss = self.mask_loss(out_mask, target_mask)
        
        if show_time:
            torch.cuda.synchronize()
            t3.record()

        if show_time:
            torch.cuda.synchronize()
            print(" PMLOSS ", t1.elapsed_time(t2), t2.elapsed_time(t3))
                
        nocs_loss = self.masked_l2_loss(pred_nocs, target_nocs, target_mask)
        
        # skin_loss = self.masked_l2_loss(self.sigmoid(pred_skin_seg), tar_skin_weights, target_mask)
        pred_seg = pred_skin_seg.transpose(1, 2).transpose(2, 3).contiguous().view(-1, bone_num+2)
        tar_seg = tar_skin_seg.long().squeeze(1).view(-1)
        # print(pred_seg.shape)
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
                
        if self.cfg.REPOSE:
            pred_pnnocs = output[:, -3:, :, :].clone().requires_grad_(True)            
            # pred_pnnocs = output[:, 4+bone_num*8+2:4+bone_num*8+5, :, :].clone().requires_grad_(True)
            tar_pnnocs = tar_maps[:, -3:, :, :]
            # tar_pnnocs = tar_maps[:, 4+bone_num*6+1:4+bone_num*6+4, :, :] #
            # assert (tar_pnnocs == tar_pnnocs_).all()
            pnnocs_loss = self.masked_l2_loss(pred_pnnocs, tar_pnnocs, target_mask)
            # loss += pnnocs_loss
            loss['pnnocs_loss'] = pnnocs_loss
            # print("[ DIFF ] map_loss is {:5f}; loc_loss is {:5f}".format(loc_map_loss, joint_loc_loss))
            

            # print("PMloss time",time_a - time_0, time_b - time_a, time_c - time_b, time_d - time_c, time_e - time_d)
            # print("PMloss time",time_e - time_0)
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
            # print("num non zero", num_non_zero)
            if num_non_zero > 0:
                l2_loss += torch.sum(masked_diff_norm[i]) / num_non_zero
            else:
                l2_loss += torch.mean(diff_norm[i])
        l2_loss /= batch_size
        # print("l2 loss ", l2_loss)
        return l2_loss

    def pose_nocs_loss(self, pred_joint_map, pred_joint_score, out_mask, tar_joints):

        n_batch = pred_joint_map.shape[0]
        bone_num = self.bone_num
        # get final prediction: score map summarize
        pred_joint_map = pred_joint_map.reshape(n_batch, bone_num, 3, pred_joint_map.shape[2],
                                                pred_joint_map.shape[3])  # B,bone_num,3,R,R
        pred_joint_map = pred_joint_map * out_mask.unsqueeze(1).unsqueeze(1)
        pred_joint_score = pred_joint_score.sigmoid() * out_mask.unsqueeze(1)
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

        if not self.config.STAGE_ONE:
            ifnet_output = output[1]
            loss['recon_loss'] = self.recon_loss(ifnet_output, target)

        all_loss = self.add_up(loss)

        return all_loss

    @staticmethod
    def recon_loss(recon, target):
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
            all_loss += cfg.NOCS_LOSS * loss['pnnocs_loss']
        if cfg.STAGE_ONE == False:
            all_loss += cfg.RECON_LOSS * loss['recon_loss']
        if cfg.CONSISTENCY != 0:
            all_loss += cfg.CONSISTENCY * loss['crr_loss']
        # print("all loss is ", all_loss)
        # if all_loss > 3:
        #     print("error, outlier")
        return all_loss
    
class MVPMLoss(PMLoss):

    def __init__(self, config):
        super().__init__(config)
        self.crr_l2 = JiahuiL2Loss()

    def forward(self, output, target):

        target.pop('mesh', None)
        target.pop('iso_mesh', None)

        crr = {}
        crr['crr-idx-mtx'] = target.pop('crr-idx-mtx', None)
        crr['crr-mask-mtx'] = target.pop('crr-mask-mtx', None)
        tar_mask = [nox[:, 3] for nox in target['nox00']]

        target_list = DL2LD(target)
        segnet_output = output[0]

        mv_loss = {}
        view_num = self.config.VIEW_NUM
        for v in range(view_num):
            if isinstance(segnet_output, list):
                sv_output = segnet_output[v]
            else:
                sv_output = segnet_output

            sv_target = target_list[v]

            tar = {}
            tar_maps = []
            for k in ['nox00', 'joint_map', 'linkseg', 'pnnocs00']:
                tar_maps.append(sv_target[k])
            tar['maps'] = torch.cat(tuple(tar_maps), dim=1)
            tar['pose'] = sv_target['pose']

            sv_loss = self.segnet_loss(sv_output, tar)

            if len(mv_loss) == 0:
                mv_loss = sv_loss
            else:
                for k in sv_loss:
                    mv_loss[k] += sv_loss[k]

        for k in mv_loss:
            mv_loss[k] /= view_num

        if not self.config.STAGE_ONE:
            ifnet_output = output[1]
            mv_loss['recon_loss'] = self.recon_loss(ifnet_output, target_list[0])
        
        if self.config.CONSISTENCY != 0:
            mv_loss['crr_loss'] = self.crr_loss(segnet_output, tar_mask, crr)

        mv_loss = self.add_up(mv_loss)



        return mv_loss

    def crr_loss(self, output_list, tar_mask, crr):        
        """
        We do this for each instance in batch
        """        
        batch_size = output_list[0].shape[0]
        crr_xyz_loss = 0
        pair_cnt = 0

        for b_id in range(batch_size):
            _p1_list, _p2_list, _m_list = [], [], []
            for base_view_id in range(len(crr['crr-idx-mtx'])):
                for query_view_id in range(len(crr['crr-idx-mtx'][base_view_id])):
                    
                    base_pn_map = output_list[base_view_id][b_id, -3:, ]
                    base_pn_map = output_list[base_view_id][b_id, -3:, ]
                    query_pn_map = output_list[base_view_id + query_view_id + 1][b_id, -3:, ]
                    base_mask = tar_mask[base_view_id][b_id]
                    query_mask = tar_mask[base_view_id + query_view_id + 1][b_id]
                    
                    base_pn_pc = base_pn_map[:, base_mask.squeeze() > 0].transpose(1, 0)
                    query_pn_pc = query_pn_map[:, query_mask.squeeze() > 0].transpose(1, 0)

                    # base_pc = pred_xyz_list[base_view_id]                    
                    # query_pc = pred_xyz_list[base_view_id + query_view_id + 1]

                    pair_idx = crr['crr-idx-mtx'][base_view_id][query_view_id][b_id]
                    paired_pc_from_base_to_query = base_pn_pc[pair_idx]
                    paired_pc_from_base_to_query = paired_pc_from_base_to_query.squeeze(0)
                    # paired_pc_from_base_to_query = torch.gather(base_pn_pc.squeeze(3), dim=2,
                    #                                             index=pair_idx.repeat(1, 3, 1)).unsqueeze(3)
                    _p1_list.append(paired_pc_from_base_to_query)
                    _p2_list.append(query_pn_pc)
                    _m_list.append(crr['crr-mask-mtx'][base_view_id][query_view_id][b_id].squeeze())
                    # print(torch.stack(_p1_list, dim=0).contiguous().shape)
                    # print(_p2_list[0].shape, _p2_list[1].shape)
                    
                    p1 = paired_pc_from_base_to_query.unsqueeze(0)
                    p2 = query_pn_pc.unsqueeze(0)
                    mask = crr['crr-mask-mtx'][base_view_id][query_view_id][b_id].squeeze()

                    if 0:
                        masked_p1 = p1[0, mask.to(dtype=bool), :].cpu().detach().numpy()
                        masked_p2 = p2[0, mask.to(dtype=bool), :].cpu().detach().numpy()
                        size = 100
                        points_num = masked_p1.shape[0]
                        print("crr num: ", points_num)
                        step = points_num // size
                        import os             
                        crr_root = "../crr"       
                        if not os.path.exists(crr_root):
                            os.mkdir(crr_root)
                        # for i in range(step):
                        #     write_off("/data/new_disk2/zhangge/crr/pred_crr_0_{}.xyz".format(i), masked_p1[i*size:(i+1)*size])
                        #     write_off("/data/new_disk2/zhangge/crr/pred_crr_1_{}.xyz".format(i), masked_p2[i*size:(i+1)*size])                    
                        crr_id = len(os.listdir(crr_root))
                        crr_dir = os.path.join(crr_root, str(crr_id))
                        if not os.path.exists(crr_dir):
                            os.mkdir(crr_dir)
                        write_off(os.path.join(crr_dir, "masked_p1.xyz"), masked_p1)
                        write_off(os.path.join(crr_dir, "masked_p2.xyz"), masked_p2)
                        write_off(os.path.join(crr_dir, "p1.xyz"), p1[0].cpu().detach().numpy())
                        write_off(os.path.join(crr_dir, "p2.xyz"), p2[0].cpu().detach().numpy())
                        write_off(os.path.join(crr_dir, "p_query.xyz"), query_pn_pc.cpu().detach().numpy())
                        # print(mask.shape)
                    crr_xyz_loss += self.crr_l2(p1, p2, mask, detach=False)
                    pair_cnt += 1                    
                    # exit()

        crr_xyz_loss /= pair_cnt

        return crr_xyz_loss

class MVMPLoss(MVPMLoss):
    
    def __init__(self, config):
        super().__init__(config)
    
    def forward(self, output, target):
        
        show_time = 0
        # show_time = 0
        target.pop('mesh', None)
        target.pop('iso_mesh', None)
        tar_mask = [nox[:, 3] for nox in target['nox00']]

        crr = {}
        crr['crr-idx-mtx'] = target.pop('crr-idx-mtx', None)
        crr['crr-mask-mtx'] = target.pop('crr-mask-mtx', None)
        
        cano_occ = target.pop('cano_occupancies') # TODO

        target_list = DL2LD(target)
        segnet_output = output[0]

        if show_time:
            torch.cuda.synchronize()
            time_0 = time()

        mv_loss = {}
        view_num = self.config.VIEW_NUM
        for v in range(view_num):
            if show_time:
                torch.cuda.synchronize()
                time_a = time()

            if isinstance(segnet_output, list):
                sv_output = segnet_output[v]
            else:
                sv_output = segnet_output

            sv_target = target_list[v]

            tar = {}
            tar_maps = []
            for k in ['nox00', 'joint_map', 'linkseg', 'pnnocs00']:
                tar_maps.append(sv_target[k])
            tar['maps'] = torch.cat(tuple(tar_maps), dim=1)
            tar['pose'] = sv_target['pose']

            sv_loss = self.segnet_loss(sv_output, tar)
        
            if len(mv_loss) == 0:
                mv_loss = sv_loss
            else:
                for k in sv_loss:
                    mv_loss[k] += sv_loss[k]
            
            if show_time:
                torch.cuda.synchronize()
                time_b = time()
                print(" time step {} {} {} {} ".format(v, time_a, time_b, time_b - time_a))

        if show_time:
            torch.cuda.synchronize()
            time_1 = time()

        for k in mv_loss:
            mv_loss[k] /= view_num
        
        # print("time all ", time_0, time_a, time_1 - time_0)
        if not self.config.STAGE_ONE:
            pn_recon = output[1]
            recon_loss = self.recon_loss(pn_recon, cano_occ)

            posed_recons = output[2]
            for i in range(view_num):
                posed_recon_loss = self.recon_loss(posed_recons[i], target_list[i]['occupancies'])
                # print(posed_recon_loss)
                recon_loss += posed_recon_loss
            
            recon_loss /= (view_num + 1)
            mv_loss['recon_loss'] = recon_loss

        if self.config.CONSISTENCY != 0:
            mv_loss['crr_loss'] = self.crr_loss(segnet_output, tar_mask, crr)
        
        
        # if show_time:
            # print("loss time",time_b - time_a, time_c - time_b)
            # print("loss time", time_1 - time_0)
        mv_loss = self.add_up(mv_loss)

        return mv_loss
        
    def recon_loss(self, recon, occ):
        
        sm = occ.sum()
        # out = (B,num_points) by componentwise comparing vecots of size num_samples :)
        occ_loss = nn.functional.binary_cross_entropy_with_logits(
                recon, occ, reduction='none')
        num_sample = occ_loss.shape[1]
        occ_loss = occ_loss.sum(-1).mean() / num_sample
        return occ_loss
