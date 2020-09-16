'''
Created by Ge on Sept. 16
'''

import torch.nn as nn
import torch.nn.functional as F
import torch 
import torch.optim as optim

import os, sys, argparse, math, glob, gc, traceback
import numpy as np
import cv2
import torch
import torch.nn as nn
from models.networks.MultiHeadSegNet import MHSegNet
from models.networks.IFNet import SVR
import utils.tools.implicit_waterproofing as iw
from utils.lbs import *

from utils.tools.voxels import VoxelGrid
import torch_scatter
from utils.tools.pc2voxel import voxelize as pc2vox
from loaders.HandOccDataset import HandOccDataset


class LNRNet(nn.Module):

    def __init__(self, config, device=torch.device("cpu"), is_unpooling=True, Args=None, pretrained=True, withSkipConnections=False,Sample=False):
        super().__init__()
        self.config = config        
        self.SegNet = MHSegNet()

        self.SegNet.to(device)
        self.device = device
        self.IFNet = SVR(config, device)
        
        self.resolution = 128
        self.initGrids(self.resolution)
                
        self.UpdateSegNet = config.UPDATE_SEG

        self.use_pretrained = False

        self.init_hp()
        # Freeze the SegNet part due to the bug
        # if self.FreezeSegNet:
        for param in self.SegNet.parameters():
            param.requires_grad = self.UpdateSegNet
    
    def init_hp(self):
        
        self.Sample = False
        self.SampleNum = 3000 # necessary?
        self.Vis = False
        self.joint_num = 16
        self.sigmoid = nn.Sigmoid()

        # index of different output
        self.nocs_end = 3
        self.mask_end = self.nocs_end + 1
        self.ft_end =   self.nocs_end + 64
        self.loc_end =  self.ft_end + 16*3
        self.pose_end = self.loc_end + 16*3
        self.skin_end = self.pose_end + 16
        self.conf_end = self.skin_end + 16

    def initGrids(self, resolution):
        bb_min = -0.5
        bb_max = 0.5

        self.GridPoints = iw.create_grid_points_from_bounds(bb_min, bb_max, resolution)
        grid_points = iw.create_grid_points_from_bounds(bb_min, bb_max, resolution)
        grid_points[:, 0], grid_points[:, 2] = grid_points[:, 2], grid_points[:, 0].copy()
        
        a = bb_max + bb_min
        b = bb_max - bb_min
        
        grid_coords = 2 * grid_points - a
        grid_coords = grid_coords / b
        grid_coords = torch.from_numpy(grid_coords).to(self.device, dtype=torch.float)
        grid_coords = torch.reshape(grid_coords, (1, len(grid_points), 3)).to(self.device)
        return grid_coords
    
    def forward(self, inputs):
        # print(type(inputs))
        color = inputs['RGB']
        # print(Color.size())
        transform = {'translation': inputs['translation'],
                     'scale':inputs['scale']}

        # Grids, comes from boundary sampling during training and a boudary cube during vlidation
        # This operation is simply for the sake of training speed
        grid_coords = inputs['grid_coords']

        # here we edit the intermediate output for the IF-Net stage
        # first lift them into 3D point cloud
        output = self.SegNet(color)
        
        pred_nocs = output[:, :self.nocs_end, :, :].clone().requires_grad_(True)
        pred_mask = output[:, self.nocs_end:self.mask_end, :, :].clone().requires_grad_(True)
        nocs_feature = output[:, self.nocs_end:self.ft_end, :, :].clone().requires_grad_(True)
        pred_loc = output[:, self.ft_end:self.loc_end, :, :].clone().requires_grad_(True)
        pred_pose = output[:, self.loc_end:self.pose_end, :, :].clone().requires_grad_(True)
        pred_weight = output[:, self.pose_end:self.skin_end, :, :].clone().requires_grad_(True)
        conf = output[:, self.skin_end:self.conf_end, :, :].clone().requires_grad_(True)

        pnnocs = self.repose(pred_nocs, pred_loc, pred_pose, pred_weight, conf, pred_mask)

        # then: we transform the point cloud into occupancy(along with the features )
        occupancies = self.voxelize(pnnocs, nocs_feature, transform)

        # if self.Vis == True:
        # self.visualize(occupancies, inputs['translation'], inputs['scale'])
        # and then feed into IF-Net. The ground truth shouled be used in the back projection
        recon = self.IFNet(grid_coords, occupancies)
        return pred_nocs, recon

    def vote(self, pred_joint_map, pred_joint_score, out_mask):
        
        n_batch = pred_joint_map.shape[0]
        bone_num = self.joint_num

        # get final prediction: score map summarize
        pred_joint_map = pred_joint_map.reshape(n_batch, bone_num, 3, pred_joint_map.shape[2],
                                                pred_joint_map.shape[3])  # B,bone_num,3,R,R
        pred_joint_map = pred_joint_map * out_mask.unsqueeze(1).unsqueeze(1)
        pred_joint_score = self.sigmoid(pred_joint_score) * out_mask.unsqueeze(1)
        pred_score_map = pred_joint_score / (torch.sum(pred_joint_score.reshape(n_batch, bone_num, -1),
                                                    dim=2, keepdim=True).unsqueeze(3) + 1e-5)

        pred_joint_map = pred_joint_map.detach() * pred_score_map.unsqueeze(2)
        pred_joint = pred_joint_map.reshape(n_batch, bone_num, 3, -1).sum(dim=3)  # B,22,3

        return pred_joint
    
    def repose(self, pred_nocs, pred_loc_map, pred_pose_map, pred_weight, conf, pred_mask):
        
        pred_loc = self.vote(pred_loc_map, conf, pred_mask)
        pred_pose = self.vote(pred_pose_map, conf, pred_mask)        
        back_pose = -pred_pose

        pixel_weights = torch.stack(pred_weight)
        pixel_weights = pixel_weights.transpose(0, 1) # Bone pixel weights now have shape NXJ
        
        # only move pixel whose dominant joint is visible
        valid_pixels = pixel_weights.sum(dim=1) > 0.99
        pixel_weights = pixel_weights[valid_pixels, :]
        nocs_pc = pred_nocs[:, valid_pixels, :]
        T, _ = rotation_matrix(back_pose, pred_loc, pixel_weights, self.parents, dtype=nocs_pc.dtype)
        pnnocs = lbs_(nocs_pc, T, dtype=nocs_pc.dtype)

        return pnnocs

        def voxelize(self, output, feature, transform):
        batch_size = output.size(0)
        img_size = (output.size(2),output.size(3))
        feature_dim = feature.shape[1]
        
        # get masked nocs
        out_mask = output[:, -1, :, :].clone().requires_grad_(True)
        sigmoid = nn.Sigmoid()
        out_mask = sigmoid(out_mask)
        threshold = 0.75
        pred_nocs = output[:, :-1, :, :].clone().requires_grad_(True)

        valid = out_mask > threshold
        masked_nocs = torch.where(torch.unsqueeze(valid, 1), pred_nocs, torch.zeros(pred_nocs.size(), device=pred_nocs.device))
        # print(masked_nocs)
        # get upsampeld feature
        upsampled_feature = F.interpolate(feature, size=img_size)

        all_occupancies = []
        for i in range(batch_size):            
            img = masked_nocs[i, :].cpu().detach().numpy()
            valid_idx = np.where(np.all(img > np.zeros((3, 1, 1)), axis=0)) # Only white BG
            # valid_idx = masked_nocs[i] > 0
            index = valid_idx

            num_valid = valid_idx[0].shape[0]
            if num_valid == 0:
                # No valid point at all. This will cut off the gradient flow
                # occ_empty = np.zeros(len(self.GridPoints), dtype=np.int8)
                # occ_empty = np.reshape(occ_empty, (self.resolution,)*3)
                occ_empty = torch.ones(feature_dim, *(self.resolution,)*3).to(device=masked_nocs.device)
                print("empty", occ_empty.shape)
                all_occupancies.append(occ_empty)
                continue

            if self.Sample:
                random_index = np.random.choice(num_valid, self.SampleNum, replace=True)
                # for current use we choose uniform sample
                sampled_idx = (valid_idx[0][random_index], valid_idx[1][random_index])
                index = sampled_idx

            pointcloud = masked_nocs[i, :, index[0], index[1]]
            
            if transform != None:
                translation = transform['translation'][i].view(3, 1).float()
                pointcloud = pointcloud + translation
                pointcloud = pointcloud * transform['scale'][i]

            # self.save_mesh(pointcloud)
            # pc_lower_bound, _ = pointcloud.min(dim=1)
            # pointcloud -= pc_lower_bound.unsqueeze(1)

            if 1:
                # Feature solution
                feature_cloud = upsampled_feature[i, :, index[0], index[1]]
                voxelized_feature = self.discretize(pointcloud, feature_cloud, self.resolution)
                all_occupancies.append(voxelized_feature)
                # print(voxelized_feature.shape)
            else:
                # occupancy solution
                c, n = pointcloud.shape
                pointcloud = pointcloud.view(1, n, c)
                voxel = pc2vox(pointcloud, self.resolution)
                all_occupancies.append(voxel)
        
        # AllOccupancies = torch.Tensor(np.array(AllOccupancies)).to(device=PointCloud.device,dtype=torch.float32)
        # print(len(AllOccupancies))
        all_occupancies = torch.stack(tuple(all_occupancies))
        
        return all_occupancies
    
    def voxelize(self, output, feature, transform):
        batch_size = output.size(0)
        img_size = (output.size(2),output.size(3))
        feature_dim = feature.shape[1]
        
        # get masked nocs
        out_mask = output[:, -1, :, :].clone().requires_grad_(True)
        sigmoid = nn.Sigmoid()
        out_mask = sigmoid(out_mask)
        threshold = 0.75
        pred_nocs = output[:, :-1, :, :].clone().requires_grad_(True)

        valid = out_mask > threshold
        masked_nocs = torch.where(torch.unsqueeze(valid, 1), pred_nocs, torch.zeros(pred_nocs.size(), device=pred_nocs.device))
        # print(masked_nocs)
        # get upsampeld feature
        upsampled_feature = F.interpolate(feature, size=img_size)

        all_occupancies = []
        for i in range(batch_size):            
            img = masked_nocs[i, :].cpu().detach().numpy()
            valid_idx = np.where(np.all(img > np.zeros((3, 1, 1)), axis=0)) # Only white BG
            # valid_idx = masked_nocs[i] > 0
            index = valid_idx

            num_valid = valid_idx[0].shape[0]
            if num_valid == 0:
                # No valid point at all. This will cut off the gradient flow
                # occ_empty = np.zeros(len(self.GridPoints), dtype=np.int8)
                # occ_empty = np.reshape(occ_empty, (self.resolution,)*3)
                occ_empty = torch.ones(feature_dim, *(self.resolution,)*3).to(device=masked_nocs.device)
                print("empty", occ_empty.shape)
                all_occupancies.append(occ_empty)
                continue

            if self.Sample:
                random_index = np.random.choice(num_valid, self.SampleNum, replace=True)
                # for current use we choose uniform sample
                sampled_idx = (valid_idx[0][random_index], valid_idx[1][random_index])
                index = sampled_idx

            pointcloud = masked_nocs[i, :, index[0], index[1]]
            
            if transform != None:
                translation = transform['translation'][i].view(3, 1).float()
                pointcloud = pointcloud + translation
                pointcloud = pointcloud * transform['scale'][i]

            # self.save_mesh(pointcloud)
            # pc_lower_bound, _ = pointcloud.min(dim=1)
            # pointcloud -= pc_lower_bound.unsqueeze(1)

            if 1:
                # Feature solution
                feature_cloud = upsampled_feature[i, :, index[0], index[1]]
                voxelized_feature = self.discretize(pointcloud, feature_cloud, self.resolution)
                all_occupancies.append(voxelized_feature)
                # print(voxelized_feature.shape)
            else:
                # occupancy solution
                c, n = pointcloud.shape
                pointcloud = pointcloud.view(1, n, c)
                voxel = pc2vox(pointcloud, self.resolution)
                all_occupancies.append(voxel)
        
        # AllOccupancies = torch.Tensor(np.array(AllOccupancies)).to(device=PointCloud.device,dtype=torch.float32)
        # print(len(AllOccupancies))
        all_occupancies = torch.stack(tuple(all_occupancies))
        
        return all_occupancies

    def discretize(self, PointCloud, FeatureCloud, Res):
        # input: N*3 pointcloud
        # output: 128**3 * F
        feature_dim = FeatureCloud.shape[0]
        point_num = PointCloud.shape[1]

        PointCloud = PointCloud + 0.5
        voxels = torch.floor(PointCloud*Res)

        index = voxels[0, :]*Res**2 + voxels[1, :]*Res + voxels[2, :]
        index = index.unsqueeze(0).to(dtype=torch.long)

        #TODO: replace the mean operation to pointnet
        # print(FeatureCloud.shape)
        # print(Index.shape)
        voxel_feature = torch_scatter.scatter(src=FeatureCloud, index=index)
        # VoxFeature = torch_scatter.segment_coo(src=FeatureCloud,index=Index,reduce='mean')
        pad_size = (0, Res**3 - voxel_feature.size(1))
        voxel_feature = F.pad(voxel_feature, pad_size, 'constant', 0)
        voxel_feature = voxel_feature.view(feature_dim, Res, Res, Res)
        # print(VoxFeature.shape)
        # exit()

        return voxel_feature
