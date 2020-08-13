import torch.nn as nn
import torch.nn.functional as F
import torch 
import torch.optim as optim

import os, sys, argparse, math, glob, gc, traceback
import numpy as np
import cv2
import torch
import torch.nn as nn
from models.networks.SegNet import SegNet
# from models.networks.say4n_SegNet import SegNet

from models.networks.SegNetNR import SegNetNR
from models.networks.IFNet import SVR
import utils.tools.implicit_waterproofing as iw
from utils.tools.voxels import VoxelGrid
import torch_scatter
from utils.tools.pc2voxel import voxelize as pc2vox
from loaders.HandOccDataset import HandOccDataset

#updated by Ge. Aug 10😎
class NRNet(nn.Module):
    def __init__(self, config, device=torch.device("cpu"), is_unpooling=True, Args=None, pretrained=True, withSkipConnections=False,Sample=False):
        super().__init__()        
        self.config = config
        if config.NRNET_TYPE == "out_feature":
            out_channels = config.OUT_CHANNELS + config.FEATURE_CHANNELS
            self.SegNet = SegNet(output_channels=out_channels)
        elif config.NRNET_TYPE == "inter_feature":
            out_channels = config.OUT_CHANNELS
            self.SegNet = SegNetNR(output_channels=out_channels)
        else:
            print("[ ERROR ] unsupported NRNet type")
            exit()

        self.SegNet.to(device)
        self.device = device
        self.IFNet = SVR(config, device)
        self.resolution = 128
        self.initGrids(self.resolution)
        self.Sample = False
        self.SampleNum = 3000 # necessary?
        self.FreezeSegNet = False

        self.Vis = False
        self.use_pretrained = False
        # Freeze the SegNet part due to the bug
        if self.FreezeSegNet:
            for param in self.SegNetNR.parameters():
                param.requires_grad = False

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
        nocs_wt_featr = self.SegNet(color)

        if self.config.NRNET_TYPE == "out_feature":
            pred_nocs = nocs_wt_featr[:, :4, :, :]
            nocs_feature = nocs_wt_featr[:, 4:, :, :]
        elif self.config.NRNET_TYPE == "inter_feature":
            pred_nocs = nocs_wt_featr[0]
            nocs_feature = nocs_wt_featr[1]
        else:
            print("[ ERROR ] unsupported NRNet type")
            exit()

        # then: we transform the point cloud into occupancy(along with the features )
        occupancies = self.voxelize(pred_nocs, nocs_feature, transform)

        if self.Vis == True:
            self.visualize(occupancies, pred_nocs)

        # and then feed into IF-Net. The ground truth shouled be used in the back projection
        recon = self.IFNet(grid_coords, occupancies)
        return pred_nocs, recon

    def voxelize(self, output, feature, transform):
        
        batch_size = output.size(0)
        img_size = (output.size(2),output.size(3))
        feature_dim = feature.shape[1]
        
        # get masked nocs
        out_mask = output[:, -1, :, :].clone().requires_grad_(True)
        Sigmoid = nn.Sigmoid()
        out_mask = Sigmoid(out_mask)
        threshold = 0.75
        pred_nocs = output[:, :-1, :, :].clone().requires_grad_(True)
        
        valid = out_mask > threshold
        masked_nocs = torch.where(torch.unsqueeze(valid, 1), pred_nocs, torch.zeros(pred_nocs.size(), device=pred_nocs.device))

        # get upsampeld feature
        upsampled_feature = F.interpolate(feature, size=img_size)
        
        all_occupancies = []
        for i in range(batch_size):

            img = masked_nocs[i, :].cpu().detach().numpy()
            valid_idx = np.where(np.all(img > np.zeros((3, 1, 1)), axis=0)) # Only white BG
            index = valid_idx

            num_valid = valid_idx[0].shape[0]
            if num_valid == 0:
                # No valid point at all. This will cut off the gradient flow
                # occ_empty = np.zeros(len(self.GridPoints), dtype=np.int8)
                # occ_empty = np.reshape(occ_empty, (self.resolution,)*3)
                occ_empty = torch.zeros(feature_dim, *(self.resolution,)*3).to(device=masked_nocs.device)
                # print("empty", occ_empty.shape)
                all_occupancies.append(occ_empty)
                continue

            if self.Sample:
                random_index = np.random.choice(num_valid, self.SampleNum, replace=True)
                # for current use we choose uniform sample
                sampled_idx = (valid_idx[0][random_index], valid_idx[1][random_index])
                index = sampled_idx

            pointcloud = masked_nocs[i, :, index[0], index[1]]
            translation = transform['translation'][i].view(3, 1).float()

            pointcloud = pointcloud + translation
            pointcloud = pointcloud * transform['scale'][i]

            pc_lower_bound, _ = pointcloud.min(dim=1)
            pointcloud -= pc_lower_bound.unsqueeze(1)

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

    def showGradient(self):
        for name, parms in self.named_parameters():
            if parms.grad is not None:
                v = parms.grad.sum()
            else:
                v = None
            print('-->name:', name, '-->grad_requirs:', parms.requires_grad, \
        	 ' -->grad_value:',v)

    def visualize(self,FeatureVoxel,NOCS):        
        FeatureSample = FeatureVoxel[0]
        Voxel = (FeatureSample != 0).sum(dim=0).to(dtype=torch.bool)
        OffPath = "/workspace/NRNet/debug/mid.off"
        VoxelGrid(Voxel.cpu().detach(), (0,0,0), 1).to_mesh().export(OffPath)
        print(NOCS.shape)
        _, PredOutTupRGB, PredOutTupMask = IFHandRigDataset.convertData(ptUtils.sendToDevice(NOCS[0].detach(), 'cpu'),\
                ptUtils.sendToDevice(NOCS.detach(), 'cpu'), isMaskNOX=True)
        
        cv2.imwrite("/workspace/NRNet/debug/predNOCS.png", cv2.cvtColor(PredOutTupRGB[0], cv2.COLOR_BGR2RGB))
        exit()
