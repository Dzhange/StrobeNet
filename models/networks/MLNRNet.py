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
from models.networks.IFNet import SVR
from models.networks.LNRNet import LNRNet

import utils.tools.implicit_waterproofing as iw
from utils.lbs import *
from utils.tools.voxels import VoxelGrid
from utils.tools.pc2voxel import voxelize as pc2vox

class MLNRNet(LNRNet):

    def __init__(self, config):
        super().__init__(config)
        self.aggr_scatter = config.AGGR_SCATTER
        self.view_num = config.VIEW_NUM

    def init_network(self):
        self.SegNet = MVTHSegNet(pose_channels=self.joint_num*(3+3+1+1)+2, bn=self.config.BN)
        self.SegNet.to(self.device)
        self.IFNet = SVR(self.config, self.device)

    def forward(self, inputs):
        
        img_list = inputs['color00']
        batch_size = self.config.BATCHSIZE
        if self.transform:
            transform = {'translation': inputs['translation'],
                     'scale':inputs['scale']}
        else:
            transform = None
        # Grids, comes from boundary sampling during training and a boudary cube during vlidation
        # This operation is simply for the sake of training speed
        grid_coords = inputs['grid_coords']
        # here we edit the intermediate output for the IF-Net stage
        # first lift them into 3D point cloud
        pred_list = self.SegNet(img_list)

        mv_pc_list = []
        mv_feature_list = []
        
        
        b_mv_occupancy_list = [] # occupancy stored in it are batch wise

        for i in range(len(img_list)):
            
            output = pred_list[i]
            pred_nocs = output[:, :self.nocs_end, :, :].clone().requires_grad_(True)
            pred_mask = output[:, self.nocs_end:self.mask_end, :, :].clone().requires_grad_(True)        
            pred_loc = output[:, self.mask_end:self.loc_end, :, :].clone().requires_grad_(True)
            pred_pose = output[:, self.loc_end:self.pose_end, :, :].clone().requires_grad_(True)
            pred_weight = output[:, self.pose_end:self.skin_end, :, :].clone().requires_grad_(True)
            conf = output[:, self.skin_end:self.conf_end, :, :].clone().requires_grad_(True)
            nocs_feature = output[:, self.conf_end:self.ft_end, :, :].clone().requires_grad_(True)
            if self.config.REPOSE:
                pnnocs_maps = self.repose_pm_pred(pred_nocs, pred_loc, pred_pose, pred_weight, conf, pred_mask)
                # pnnocs_maps = self.repose_pm(pred_nocs, pred_loc, pred_pose, pred_weight, conf, pred_mask)
                output = torch.cat((output, pnnocs_maps), dim=1)
            recon = None
            if not self.config.STAGE_ONE:
                # then: we transform the point cloud into occupancy(along with the features )
                sv_pc_list, sv_feature_list = self.lift(output, nocs_feature, transform)

                if self.aggr_scatter:
                    # this list is batch wise                
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
        
        if self.aggr_scatter:
            occupancy_list = []
            for i in range(batch_size):
                # we aggregate for each item in the batch
                batch_pc_list = []
                batch_feature_list = []
                for view in range(self.view_num):
                    batch_pc_list.extend(mv_pc_list[view][i])
                    batch_feature_list.extend(mv_feature_list[view][i])
                batch_pc = torch.stack(tuple(batch_pc_list))
                batch_feature = torch.stack(tuple(batch_feature_list))

                occupancy = self.discretize(batch_pc, batch_feature, self.resolution)
                occupancy_list.append(occupancy)
            occupancies = torch.stack(tuple(occupancy_list))
        else:
            mv_occupancy = torch.stack(tuple(b_mv_occupancy_list), dim=1)
            mv_occupancy = self.avgpool_grids(mv_occupancy)

        # and then feed into IF-Net. The ground truth shouled be used in the back projection
        recon = self.IFNet(grid_coords, occupancies)
        return output, recon
    
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
