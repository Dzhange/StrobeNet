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


    def init_network(self):
        self.SegNet = MVTHSegNet(pose_channels=self.joint_num*(3+3+1+1)+2, bn=self.config.BN)
        self.SegNet.to(self.device)
        self.IFNet = SVR(self.config, self.device)

    def forward(self, inputs):
        
        img_list = inputs['color00']

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
        occupancy_list = []
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
                occupancies = self.voxelize(output, nocs_feature, transform)
                occupancy_list.append(occupancies)
                
                if 0:
                    self.visualize(occupancies)
                # and then feed into IF-Net. The ground truth shouled be used in the back projection

        recon = self.IFNet(grid_coords, occupancies)
        return output, recon
        

