import os, sys, argparse, math, glob, gc, traceback
import numpy as np
import transforms3d

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torch_scatter
from models.networks.MVTHSegNet import MVTHSegNet
from models.networks.TripleHeadSegNet import THSegNet
from models.networks.IFNet import SVR, SuperRes
from models.networks.ShallowIFNet import ShallowSVR
from models.networks.LNRNet import LNRNet
from time import time
import utils.tools.implicit_waterproofing as iw
from utils.lbs import *
from utils.tools.voxels import VoxelGrid
from utils.tools.pc2voxel import voxelize as pc2vox
from utils.DataUtils import DL2LD, LD2DL

class MLNRNet(LNRNet):

    def __init__(self, config, device):
        super().__init__(config, device=device)
        self.aggr_scatter = config.AGGR_SCATTER
        self.view_num = config.VIEW_NUM

    def init_network(self):
        self.SegNet = MVTHSegNet(pose_channels=self.joint_num*(3+3+1+1)+2, \
            feature_channels=self.config.FEATURE_CHANNELS, pred_feature=self.config.PRED_FEATURE,\
            bn=self.config.BN, return_code=self.config.GLOBAL_FEATURE)
        # self.SegNet = THSegNet(pose_channels=self.joint_num*(3+3+1+1)+2, bn=self.config.BN)
        self.SegNet.to(self.device)
        if self.config.IF_SHALLOW:
            self.IFNet = ShallowSVR(self.config, self.device)
        if self.config.SUPER_RES:
            self.IFNet = SuperRes(self.config, self.device)
        else:
            self.IFNet = SVR(self.config, self.device)
        
        if self.config.GLOBAL_FEATURE:
            # from Jiahui
            self.concentrate = nn.Sequential(
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=0, stride=1),
                nn.BatchNorm2d(512),
                nn.ELU(),
                nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=0, stride=1),
                nn.MaxPool2d(kernel_size=(3, 6))
            )
        # if not self.config.PRED_FEATURE:
            # torch.backends.cudnn.enabled = False

    def forward(self, inputs):

        img_list = inputs['color00']
        batch_size = img_list[0].shape[0]

        # DEBUG: try old network
        
        if isinstance(self.SegNet, MVTHSegNet):
            if self.config.GLOBAL_FEATURE:
                pred_list, featuremap_list = self.SegNet(img_list)
                code_list = []
                for fm in featuremap_list:  # do for each view
                    code_list.append(self.concentrate(fm).reshape(batch_size, -1, 1).contiguous())
                global_z = torch.max(torch.cat(code_list, 2), dim=2).values.contiguous()
                # print(global_z.shape)
            else:
                pred_list = self.SegNet(img_list)
            
        else:
            sv_output = self.SegNet(img_list[0])            
    
        mv_pc_list = []
        mv_feature_list = []
        b_mv_occupancy_list = [] # occupancy stored in it are batch wise

        if self.transform:
            transform = {'translation': inputs['translation'],
                     'scale':inputs['scale']}
            transform_list = DL2LD(transform)
        else:
            transform_list = None

        output_list = []        
        for i in range(len(img_list)):
            if isinstance(self.SegNet, MVTHSegNet):
                sv_output = pred_list[i]
            # else:
                # sv_output = pred_item
            sv_output, nocs_feature = self.process_sv(sv_output)
            output_list.append(sv_output)

            if not self.config.STAGE_ONE:
                # sv_pc_list, sv_feature_list contrains pc from the same view_id in all batches
                sv_pc_list, sv_feature_list = self.lift(sv_output, nocs_feature, transform_list[i])
                if self.aggr_scatter:
                    # this list is batch wise
                    if sv_pc_list is not None and mv_feature_list is not None:
                        mv_pc_list.append(sv_pc_list)
                        mv_feature_list.append(sv_feature_list)
                else:
                    b_sv_occupancy_list = []
                    for i in range(batch_size):
                        occupancy = self.discretize(sv_pc_list[i], sv_feature_list[i], self.resolution)
                        b_sv_occupancy_list.append(occupancy)
                    b_sv_occupancy = torch.stack(tuple(b_sv_occupancy_list))
                    b_mv_occupancy_list.append(b_sv_occupancy)
        
        ##### aggregation #####
        
        recon = None
        if not self.config.STAGE_ONE:
            if isinstance(inputs['grid_coords'], list):
                grid_coords = inputs['grid_coords'][0] # grid coords should be identical among all views
            else:
                grid_coords = inputs['grid_coords'] # some times(in validation time, we feed it with one tensor)
        if not self.config.STAGE_ONE:
            if self.aggr_scatter:
                occupancy_list = []
                for i in range(batch_size):
                    # we aggregate for each item in the batch
                    instance_pc, instance_feature = self.collect_pc(mv_pc_list, mv_feature_list, batch_id=i)                    
                    # write_off("/workspace/test.xyz", instance_pc.transpose(1,0))
                    if self.config.USE_FEATURE:
                        occupancy = self.discretize(instance_pc, instance_feature, self.resolution)
                    else:
                        occupancy = self.discretize_no_feature(instance_pc, self.resolution)
                        # occupancy = occupancy.unsqueeze(dim=0)
                    occupancy_list.append(occupancy)
                occupancies = torch.stack(tuple(occupancy_list))
            else:
                mv_occupancy = torch.stack(tuple(b_mv_occupancy_list), dim=1)
                mv_occupancy = self.avgpool_grids(mv_occupancy)

            # self.visualize(occupancies)
            # and then feed into IF-Net. The ground truth shouled be used in the back projection
            if self.config.GLOBAL_FEATURE:
                recon = self.IFNet(grid_coords, occupancies, global_z)
            else:
                recon = self.IFNet(grid_coords, occupancies)
        
        # return output_list, recon
        if isinstance(self.SegNet, MVTHSegNet):
            return output_list, recon
        else:
            return sv_output, recon

    def process_sv(self, sv_output):
        pred_nocs = sv_output[:, :self.nocs_end, :, :].clone().requires_grad_(True)
        pred_mask = sv_output[:, self.nocs_end:self.mask_end, :, :].clone().requires_grad_(True)
        pred_loc = sv_output[:, self.mask_end:self.loc_end, :, :].clone().requires_grad_(True)
        pred_pose = sv_output[:, self.loc_end:self.pose_end, :, :].clone().requires_grad_(True)
        pred_seg = sv_output[:, self.pose_end:self.skin_end, :, :].clone().requires_grad_(True)
        conf = sv_output[:, self.skin_end:self.conf_end, :, :].clone().requires_grad_(True)
        nocs_feature = sv_output[:, self.conf_end:self.ft_end, :, :].clone().requires_grad_(True)
        
        if self.config.REPOSE:
            # b_f = time()
            pnnocs_maps = self.repose_pm_pred(pred_nocs, pred_loc, pred_pose, pred_seg, conf, pred_mask)
            # print("function time ", time() - b_f)
            # pnnocs_maps = self.repose_pm(pred_nocs, pred_loc, pred_pose, pred_weight, conf, pred_mask)
            sv_output = torch.cat((sv_output, pnnocs_maps), dim=1)

        return sv_output, nocs_feature

    def collect_pc(self, mv_pc_list, mv_feature_list, batch_id):
        # we aggregate for each item in the batch
        instance_pc_list = []
        instance_feature_list = []
        valid_view_num = len(mv_pc_list)
        for view in range(valid_view_num):
            # print(len(mv_pc_list[view]))
            cur_pc = mv_pc_list[view][batch_id]
            cur_feature = mv_feature_list[view][batch_id]
            if cur_pc is not None and cur_feature is not None:
                instance_pc_list.append(cur_pc)
                instance_feature_list.append(cur_feature)
        if len(instance_pc_list) == 0 or len(instance_feature_list) == 0:
            instance_pc = None
            instance_feature = None
        else:
            instance_pc = torch.cat(tuple(instance_pc_list), dim=1)
            instance_feature = torch.cat(tuple(instance_feature_list), dim=1)
        return instance_pc, instance_feature

    def avgpool_grids(self, feature_grids):
        """
        Copied from Srinath
        """
        # # TODO: There is a CUDA bug in max_pool1d, so using avgSubtract
        # return self.avgSubtract(FeatureMap)
        # print('-'*50)
        # print('FeatureMap:', FeatureMap.size())
        B, S, C, Res, _, _ = feature_grids.size()
        feature_grids_p = feature_grids.view(B, S, -1)
        # print('View:', FeatureMap_p.size())
        feature_grids_p = feature_grids_p.permute(0, 2, 1) # Set size should be the last dimension
        # print('Permuted:', FeatureMap_p.size())
        # print('S:', S)
        # MP = FeatureMap_p[:, :, 0].unsqueeze(2)  # TEMP TESTING TODO
        MP = F.max_pool1d(feature_grids_p, S)
        # MP = self.MaxPool1D(FeatureMap_p)
        # print('MaxPooled:', MP.size())
        MP = MP.permute(0, 2, 1) # Undo previous permute
        # print('Permuted:', MP.size())
        MP = MP.view(B, 1, C, Res, Res, Res)
        MP.squeeze(dim=1)

        return MP

