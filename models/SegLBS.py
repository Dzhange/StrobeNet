import os
import glob
import torch
import torch.nn as nn
from models.LBS import ModelLBSNOCS
from models.networks.SegNet import SegNet as old_SegNet
from models.networks.say4n_SegNet import SegNet as new_SegNet
from models.loss import LBSLoss
from utils.DataUtils import *

class ModelSegLBS(ModelLBSNOCS):

    def __init__(self, config):
        super().__init__(config)
        self.label_color = np.asarray(
            [                
                [128, 0, 0],
                [0, 128, 0],
                [128, 128, 0],
                [0, 0, 128],
                [128, 0, 128],
                [0, 128, 128],
                [128, 128, 128],
                [64, 0, 0],
                [192, 0, 0],
                [64, 128, 0],
                [192, 128, 0],
                [64, 0, 128],
                [192, 0, 128],
                [64, 128, 128],
                [192, 128, 128],
                [0, 64, 0]
            ]
        )

    def label2map(self, bw):
        all_one = torch.ones(1, device=bw.device)
        all_zero = torch.zeros(1, device=bw.device)
        cated = ()
        for i in range(0, self.bone_num+2): # just for sapien
            cur_seg = torch.where(bw == i, all_one, all_zero)
            cated = cated + (cur_seg, )
        return torch.cat(cated, 0)

    def map2seg(self, bw):
        # 1, bone_num, W, H
        bw = bw.cpu()
        _, max_idx = bw.max(dim=0, keepdim=True)        
        cur_seg = torch.zeros(3, bw.shape[1], bw.shape[2])        
        for i in range(0, self.bone_num+2):
            cur_color = torch.Tensor(self.label_color[i]).unsqueeze(1).unsqueeze(2)
            cur_seg = torch.where(max_idx == i, cur_color, cur_seg)
        return cur_seg


    def save_mask(self, output, target, i):
        bone_num = self.bone_num
        mask = target[:, 3, :, :].cpu().detach()
        tar_seg = target[:, 4+bone_num*6, :, :].cpu().detach()
        gt_bw = self.label2map(tar_seg)
        # print(np.unique(gt_bw))
        gt_seg = self.map2seg(gt_bw)

        zero_map = torch.zeros(gt_seg.size())
        gt_seg = torch.where(mask > 0.7, gt_seg, zero_map)
        gt_seg = torch2np(gt_seg)

        pred_bw = output[0, 4+bone_num*6:4+bone_num*7+2, :, :]*255
        pred_seg = self.map2seg(pred_bw)
        
        # print(np.unique(pred_seg.cpu().detach().numpy()))
        pred_seg = torch.where(mask > 0.7, pred_seg, zero_map)
        pred_seg = torch2np(pred_seg)
        
        cv2.imwrite(os.path.join(self.output_dir, 'frame_{}_seg_gt.png').format(str(i).zfill(3)), gt_seg)
        cv2.imwrite(os.path.join(self.output_dir, 'frame_{}_seg_pred.png').format(str(i).zfill(3)), pred_seg)
        

    def save_mask_(self, output, target, i):
        bone_num = self.bone_num
        mask = target[:, 3, :, :]
        # sigmoid = torch.nn.Sigmoid()
        zero_map = torch.zeros(mask.size(), device=mask.device)
        pred_bw_index = 4+bone_num*6

        tar_seg = target[:, 4+bone_num*6, :, :].cpu().detach()
        gt_seg = self.label2map(tar_seg)

        for b_id in range(bone_num):
            # pred_bw = sigmoid(output[:, pred_bw_index + b_id, :, :])*255
            pred_bw = output[:, pred_bw_index + b_id, :, :]*255
            pred_bw = torch.where(mask > 0.7, pred_bw, zero_map)
            pred_bw = pred_bw.squeeze().cpu().detach().numpy()

            tar_bw = gt_seg[b_id, :, :]*255
            tar_bw = torch.where(mask.cpu() > 0.7, tar_bw, zero_map.cpu())
            tar_bw = tar_bw.squeeze().cpu().detach().numpy()

            cv2.imwrite(os.path.join(self.output_dir, 'frame_{}_BW{}_00gt.png').format(str(i).zfill(3), b_id), tar_bw)
            cv2.imwrite(os.path.join(self.output_dir, 'frame_{}_BW{}_01pred.png').format(str(i).zfill(3), b_id), pred_bw)