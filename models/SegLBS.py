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
        self.bone_num = 16

    def label2map(self, bw):
        
        all_one = torch.ones(1, bw.shape[1], bw.shape[2])
        all_zero = torch.zeros(1, bw.shape[1], bw.shape[2])
        cated = ()
        for i in range(self.bone_num):
            cur_seg = torch.where(bw == i, all_one, all_zero)
            cated = cated + (cur_seg, )

        return torch.cat(cated, 0)

    def save_mask(self, output, target, i):
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
