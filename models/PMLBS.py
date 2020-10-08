import os
import glob
import numpy as np
import torch
import torch.nn as nn
from models.SegLBS import ModelSegLBS
from models.networks.MultiHeadSegNet import MHSegNet
from models.networks.SegNet import SegNet as old_SegNet
from utils.DataUtils import *

class PMLBS(ModelSegLBS):
    """
    The network trains for partial mobility objects
    """
    def __init__(self, config):
        super().__init__(config)
        # self.bone_num = config.BONE_NUM # eyeglasses
        # self.net = MHSegNet(bn=False, pose_channels=self.bone_num*(3+3+1+1))
        # self.net = old_SegNet(output_channels=config.OUT_CHANNELS, bn=False)

    @staticmethod
    def preprocess(data, device):
        """
        put data onto the right device
        data input:
            0: color image
            1: loaded maps for supervision:
                1) segmap
                2) joint map
            2: Pose in np format
            3: occupancies
            4: GT mesh path
        """
        data_to_device = []
        for item in data:
            tuple_or_tensor = item
            tuple_or_tensor_td = item
            if isinstance(tuple_or_tensor_td, list):
                for ctr in range(len(tuple_or_tensor)):
                    if isinstance(tuple_or_tensor[ctr], torch.Tensor):
                        tuple_or_tensor_td[ctr] = tuple_or_tensor[ctr].to(device)
                    else:
                        tuple_or_tensor_td[ctr] = tuple_or_tensor[ctr]
                data_to_device.append(tuple_or_tensor_td)
            elif isinstance(tuple_or_tensor_td, (dict)):
                dict_td = {}
                keys = item.keys()
                for key in keys:
                    if isinstance(item[key], torch.Tensor):
                        dict_td[key] = item[key].to(device)
                data_to_device.append(dict_td)
            elif isinstance(tuple_or_tensor, torch.Tensor):
                tuple_or_tensor_td = tuple_or_tensor.to(device)
                data_to_device.append(tuple_or_tensor_td)
            else:
                # for gt mesh
                continue
        
        inputs = data_to_device[0]
        # TODO: uncomment these when pipeline finished
        # inputs = {}
        # inputs['RGB'] = data_to_device[0]
        # inputs['grid_coords'] = data_to_device[3]['grid_coords']
        # inputs['translation'] = data_to_device[3]['translation']
        # inputs['scale'] = data_to_device[3]['scale']

        targets = {}
        targets['maps'] = data_to_device[1]
        targets['pose'] = data_to_device[2]
        targets['occupancies'] = data_to_device[3]['occupancies']
        targets['mesh'] = data[4]
        
        return inputs, targets

    def label2map(self, bw):
        all_one = torch.ones(1, bw.shape[1], bw.shape[2])
        all_zero = torch.zeros(1, bw.shape[1], bw.shape[2])
        cated = ()        
        for i in range(self.bone_num+2):
            cur_seg = torch.where(bw == i, all_one, all_zero)
            cated = cated + (cur_seg, )
        return torch.cat(cated, 0)

    def map2seg(self, bw):
        # 1, bone_num, W, H
        bw = bw.cpu()
        _, max_idx = bw.max(dim=0, keepdim=True)
        cur_seg = torch.zeros(3, bw.shape[1], bw.shape[2])
        for i in range(self.bone_num+2):
            cur_color = torch.Tensor(self.label_color[i]).unsqueeze(1).unsqueeze(2)
            cur_seg = torch.where(max_idx == i, cur_color, cur_seg)
        return cur_seg

    def save_mask(self, output, target, i):
        bone_num = self.bone_num
        mask = target[:, 3, :, :].cpu().detach()
        
        gt_seg = target[:, 4+bone_num*6, :, :].cpu().detach()
        gt_seg = self.label2map(gt_seg)
        gt_seg = self.map2seg(gt_seg)
        zero_map = torch.zeros(gt_seg.size())
        # gt_seg = torch.where(mask > 0.7, gt_seg, zero_map)
        gt_seg = torch2np(gt_seg)

        pred_seg = output[0, 4+bone_num*6:4+bone_num*7+2, :, :]
        pred_seg = self.map2seg(pred_seg)
        # print(np.unique(pred_seg.cpu().detach().numpy()))
        # pred_seg = torch.where(mask > 0.7, pred_seg, zero_map)
        pred_seg = torch2np(pred_seg)
        
        cv2.imwrite(os.path.join(self.output_dir, 'frame_{}_gt_seg.png').format(str(i).zfill(3)), gt_seg)
        cv2.imwrite(os.path.join(self.output_dir, 'frame_{}_pred_seg.png').format(str(i).zfill(3)), pred_seg)