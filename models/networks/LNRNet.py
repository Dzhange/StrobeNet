'''
Created by Ge on Sept. 16
'''
import os, sys, argparse, math, glob, gc, traceback
import numpy as np
import transforms3d

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from time import time
import torch_scatter
from models.networks.TripleHeadSegNet import THSegNet
from models.networks.IFNet import SVR
import utils.tools.implicit_waterproofing as iw
from utils.lbs import *
from utils.tools.voxels import VoxelGrid
from utils.tools.pc2voxel import voxelize as pc2vox


class LNRNet(nn.Module):

    def __init__(self, config, device=torch.device("cpu")):
        super().__init__()
        self.config = config
        self.joint_num = config.BONE_NUM
        self.device = device
        
        self.init_network()

        self.transform = self.config.TRANSFORM
        
        self.resolution = self.config.RESOLUTION
        self.init_grids(self.resolution)
        self.UpdateSegNet = config.UPDATE_SEG
        self.use_pretrained = False
        self.init_hp()
        
        # Freeze the SegNet part due to the bug
        # if self.FreezeSegNet:
        for param in self.SegNet.parameters():
            param.requires_grad = self.UpdateSegNet
    
    def init_network(self):
        self.SegNet = THSegNet(pose_channels=self.joint_num*(3+3+1+1)+2, \
            feature_channels=self.config.FEATURE_CHANNELS, pred_feature=self.config.PRED_FEATURE,\
            bn=self.config.BN)            
        self.SegNet.to(self.device)
        self.IFNet = SVR(self.config, self.device)

    def init_hp(self):
        self.sample = True        
        self.max_point = 30000
        self.vis = False
        self.sigmoid = nn.Sigmoid()

        # index of different output
        self.nocs_end = 3
        self.mask_end = self.nocs_end + 1
        self.loc_end = self.mask_end + self.joint_num*3
        self.pose_end = self.loc_end + self.joint_num*3
        self.skin_end = self.pose_end + self.joint_num + 2 # for segmentation
        self.conf_end = self.skin_end + self.joint_num
        self.ft_end = self.conf_end + self.config.FEATURE_CHANNELS

    def init_grids(self, resolution):
        if self.transform:
            bb_min = -0.5
            bb_max = 0.5
        else:
            bb_min = 0
            bb_max = 1

        # self.GridPoints = iw.create_grid_points_from_bounds(bb_min, bb_max, resolution)
        grid_points = iw.create_grid_points_from_bounds(bb_min, bb_max, resolution)
        grid_points[:, 0], grid_points[:, 2] = grid_points[:, 2], grid_points[:, 0].copy()        
        if self.transform:
            a = bb_max + bb_min # 0, 1
            b = bb_max - bb_min # 1, 1
            grid_coords = 2 * grid_points - a # 
            grid_coords = grid_coords / b
        else:
            grid_coords = grid_points

        grid_coords = torch.from_numpy(grid_coords).to(self.device, dtype=torch.float)
        grid_coords = torch.reshape(grid_coords, (1, len(grid_points), 3)).to(self.device)
        self.grid_coords = grid_coords
        return grid_coords
    
    def forward(self, inputs):
        color = inputs['color00']
        
        output = self.SegNet(color)
        batch_size = output.shape[0]

        pred_nocs = output[:, :self.nocs_end, :, :].clone().requires_grad_(True)
        pred_mask = output[:, self.nocs_end:self.mask_end, :, :].clone().requires_grad_(True)        

        pred_loc = output[:, self.mask_end:self.loc_end, :, :].clone().requires_grad_(True)
        pred_pose = output[:, self.loc_end:self.pose_end, :, :].clone().requires_grad_(True)
        pred_weight = output[:, self.pose_end:self.skin_end, :, :].clone().requires_grad_(True)
        conf = output[:, self.skin_end:self.conf_end, :, :].clone().requires_grad_(True)
        nocs_feature = output[:, self.conf_end:self.ft_end, :, :].clone().requires_grad_(True)

        if self.config.REPOSE:
            pnnocs_maps = self.repose_pm_pred(pred_nocs, pred_loc, pred_pose, pred_weight, conf, pred_mask)        
            output = torch.cat((output, pnnocs_maps), dim=1)

        recon = None
        grid_coords = inputs['grid_coords']
        if self.transform:
            transform = {'translation': inputs['translation'],
                     'scale':inputs['scale']}
        else:
            transform = None
        if not self.config.STAGE_ONE:
            # then: we transform the point cloud into occupancy(along with the features)            
            point_cloud_list, feature_cloud_list = self.lift(output, nocs_feature, transform)
            occupancy_list = []
            for i in range(batch_size):
                occupancy = self.discretize(point_cloud_list[i], feature_cloud_list[i], self.resolution)
                occupancy_list.append(occupancy)
            occupancies = torch.stack(tuple(occupancy_list))
            # if self.vis == True:
            if 0:
                self.visualize(occupancies)
            # and then feed into IF-Net. The ground truth shouled be used in the back projection

            
            recon = self.IFNet(grid_coords, occupancies)
        return output, recon

    def vote(self, pred_joint_map, pred_joint_score, out_mask):
        
        n_batch = pred_joint_map.shape[0]
        joint_num = self.joint_num
        # get final prediction: score map summarize
        pred_joint_map = pred_joint_map.reshape(n_batch, joint_num, 3, pred_joint_map.shape[2],
                                                pred_joint_map.shape[3])  # B,joint_num,3,R,R
        pred_joint_map = pred_joint_map * out_mask.unsqueeze(1).unsqueeze(1)
        pred_joint_score = pred_joint_score.sigmoid() * out_mask.unsqueeze(1)
        pred_score_map = pred_joint_score / (torch.sum(pred_joint_score.reshape(n_batch, joint_num, -1),
                                                    dim=2, keepdim=True).unsqueeze(3) + 1e-5)                
        pred_joint_map = pred_joint_map.detach() * pred_score_map.unsqueeze(2)
        pred_joint = pred_joint_map.reshape(n_batch, joint_num, 3, -1).sum(dim=3)  # B,22,3
        return pred_joint

    def repose_pm(self, pred_nocs, pred_loc_map, pred_pose_map, pred_skin, conf, pred_mask):
        """
        reposing function for partial mobility models
        """
        sigmoid = nn.Sigmoid()
        pred_mask = sigmoid(pred_mask).squeeze(1)
        threshold = 0.7
        valid = pred_mask > threshold
        batch_size = pred_mask.shape[0]
        # masked_nocs = torch.where(torch.unsqueeze(valid, 1),\
        #                         pred_nocs,\
        #                         torch.zeros(pred_nocs.size(), device=pred_nocs.device)
        #                         )

        pred_locs = self.vote(pred_loc_map, conf, pred_mask)
        pred_poses = self.vote(pred_pose_map, conf, pred_mask) # axis-angle representation for the joint

        all_pnnocs = []
        for i in range(batch_size):
            # cur_masked_nocs = masked_nocs[i, :].cpu().detach().numpy()
            # valid_idx = np.where(np.all(cur_masked_nocs > np.zeros((3, 1, 1)), axis=0))
            # index = valid_idx
            # num_valid = valid_idx[0].shape[0]
            masked = pred_mask[i] > threshold
            nocs_pc = pred_nocs[i, :, masked].transpose(0, 1).unsqueeze(0)
            seg_pc = pred_skin[i, :, masked].transpose(0, 1)
            num_valid = nocs_pc.shape[1]
            if num_valid == 0:
                # No valid point at all. This will cut off the gradient flow
                all_pnnocs.append(torch.zeros(pred_nocs[0].size()).to(device=pred_nocs.device))
                continue
            
            to_cat = ()
            _, max_idx = seg_pc.max(dim=1, keepdim=True)
            seg_flags = range(1, self.joint_num+2) # 0 is background            
            for flag in seg_flags:
                link = torch.where(max_idx == flag, torch.ones(1).to(device=pred_nocs.device), torch.zeros(1).to(device=pred_nocs.device))                
                to_cat = to_cat + (link, )
            seg_pc = torch.cat(to_cat, dim=1)

            pred_loc = pred_locs[i].unsqueeze(0)
            pred_pose = pred_poses[i].unsqueeze(0)

            # TODO: following 2 rows would be deleted
            # as here link 2 is the lens with no pose, but we didn't record that
            pred_loc = F.pad(pred_loc, (0, 0, 1, 0), value=0)
            pred_pose = F.pad(pred_pose, (0, 0, 1, 0), value=0)

            joint_num = self.joint_num + 1 #TODO
            # rotation
            rodrigues = batch_rodrigues(
                    -pred_pose.view(-1, 3),
                    dtype=pred_pose.dtype
                    ).view([-1, 3, 3])
            I_t = torch.Tensor([0, 0, 0]).to(device=pred_pose.device)\
                        .repeat((joint_num), 1).view(-1, 3, 1)
            rot_mats = transform_mat(
                            rodrigues,
                            I_t,
                            ).reshape(-1, joint_num, 4, 4)
            # translation
            I_r = torch.eye(3).to(device=pred_pose.device)\
                        .repeat(joint_num, 1).view(-1, 3, 3)
            trslt_mat = transform_mat(
                            I_r,
                            pred_loc.reshape(-1, 3, 1),
                            ).reshape(-1, joint_num, 4, 4)
            back_trslt_mat = transform_mat(
                            I_r,
                            -pred_loc.reshape(-1, 3, 1),
                            ).reshape(-1, joint_num, 4, 4)
            # whole transformation point cloud
            repose_mat = torch.matmul(
                            trslt_mat,
                            torch.matmul(rot_mats, back_trslt_mat)
                            )
            T = torch.matmul(seg_pc, repose_mat.view(1, joint_num, 16)) \
                .view(1, -1, 4, 4)
            pnnocs_pc = lbs_(nocs_pc, T, dtype=nocs_pc.dtype).to(device=pred_nocs.device)
            
            # re-normalize
            low_bound = pnnocs_pc.min(axis=1)[0]
            up_bound = pnnocs_pc.max(axis=1)[0]
            scale = (up_bound - low_bound).max()
            if scale != 0:
                pnnocs_pc -= low_bound
                pnnocs_pc /= scale

            pnnocs_map = torch.zeros(pred_nocs[0].size()).to(device=pred_nocs.device)
            pnnocs_map[:, masked] = pnnocs_pc.transpose(2, 1)
            all_pnnocs.append(pnnocs_map)

        pnnocs_maps = torch.stack(tuple(all_pnnocs))
        return pnnocs_maps

    def repose_pm_pred(self, pred_nocs, pred_loc_map, pred_pose_map, pred_seg, conf, pred_mask):
        """
        reposing function for partial mobility models
        """        
        
        pred_mask = pred_mask.sigmoid().squeeze(1)
        pred_loc = self.vote(pred_loc_map, conf, pred_mask)
        pred_rot = self.vote(pred_pose_map, conf, pred_mask) # axis-angle representation for the joint        
        
        all_pnnocs = []
        batch_size = pred_mask.shape[0]
        for i in range(batch_size):
            nocs = pred_nocs[i].clone().requires_grad_(True)
            loc = pred_loc[i].clone().requires_grad_(True)
            rot = pred_rot[i].clone().requires_grad_(True)
            seg = pred_seg[i].clone().requires_grad_(True)
            mask = pred_mask[i].clone().requires_grad_(True)
          
            # empty = torch.isnan(nocs).any() or torch.isnan(seg).any() or torch.isnan(mask).any()        
            if 0:
                all_pnnocs.append(torch.zeros(pred_nocs[0].size()).to(device=pred_nocs.device))
            else:
                # m = mask.copy()                
                pnnocs, pnnocs_map = self.repose_pm_core(nocs, loc, rot, seg, mask, self.joint_num)
                # _, pnnocs_map = self.repose_pm_fast(nocs, loc, rot, seg, mask, self.joint_num)
                if pnnocs_map is None:
                    all_pnnocs.append(torch.zeros(pred_nocs[0].size()).to(device=pred_nocs.device))
                else:
                    all_pnnocs.append(pnnocs_map)
        # print("func time: ", time() - start_func)
        pnnocs_maps = torch.stack(tuple(all_pnnocs))
        return pnnocs_maps

    @staticmethod
    def repose_pm_core(NOX, loc, rot, seg, mask, joint_num):
        """
        input shape:
            NOX: 3, H, W
            loc: N, 3
            rot: N, 3
            seg: N+2, H, W
            mask: H, W
        """
        
        thresh = 0.7
        masked = mask > thresh
        # t1 = time()        
        NOX_pc = NOX[:, masked]
        seg_pc = seg[:, masked]
        # t2 = time()
        NOX_pc = NOX_pc.transpose(0, 1)
        seg_pc = seg_pc.transpose(0, 1)
        # t3 = time()                        
        # print(t2-t1, t3-t2)    
        
        pnnocs_pc = LNRNet.repose_pc(NOX_pc, seg_pc, loc, rot, joint_num)
        if pnnocs_pc is None:
            return None, None
        # # re-normalize
        # low_bound = pnnocs_pc.min(axis=1)[0]
        # up_bound = pnnocs_pc.max(axis=1)[0]
        # scale = (up_bound - low_bound).max()

        # if scale != 0:
        #     pnnocs_pc -= low_bound
        #     pnnocs_pc /= scale
        # print(pnnocs_pc)
        pnnocs_map = torch.zeros(NOX.size(), device=NOX.device)
        pnnocs_map[:, masked] = pnnocs_pc.transpose(2, 1)

        return pnnocs_pc, pnnocs_map

    @staticmethod
    def repose_pc(NOX_pc, seg_pc, loc, rot, joint_num):
        num_valid = NOX_pc.shape[0]
        if num_valid == 0:
            # No valid point at all. This will cut off the gradient flow
            return None
        to_cat = ()
        # using max_idx to confirm the segmentation
        _, max_idx = seg_pc.max(dim=1, keepdim=True)        
        
        start_rp = time()
        seg_flags = range(1, joint_num+2) # 0 is background
        for flag in seg_flags:
            part = (max_idx == flag)
            link = torch.where(part, torch.ones(1, device=NOX_pc.device), torch.zeros(1, device=NOX_pc.device))
            to_cat = to_cat + (link, )        
        
        seg_pc = torch.cat(to_cat, dim=1)
        loc = loc.unsqueeze(0)
        rot = rot.unsqueeze(0)
        # print(loc)
        # TODO: following 2 rows would be deleted
        # as here link 2 is the lens with no pose, but we didn't record that
        # print(loc, rot)
        loc = F.pad(loc, (0, 0, 1, 0), value=0)
        rot = F.pad(rot, (0, 0, 1, 0), value=0)

        # we will add the base joint, it's identical
        joint_num = joint_num + 1
        
        # rotation
        rodrigues = batch_rodrigues(
                -rot.view(-1, 3),
                dtype=rot.dtype
                ).view([-1, 3, 3])
        I_t = torch.Tensor([0, 0, 0]).to(device=rot.device)\
                    .repeat((joint_num), 1).view(-1, 3, 1)
        rot_mats = transform_mat(
                        rodrigues,
                        I_t,
                        ).reshape(-1, joint_num, 4, 4)

        # translation
        I_r = torch.eye(3).to(device=rot.device)\
                    .repeat(joint_num, 1).view(-1, 3, 3)
        trslt_mat = transform_mat(
                        I_r,
                        loc.reshape(-1, 3, 1),
                        ).reshape(-1, joint_num, 4, 4)
        back_trslt_mat = transform_mat(
                        I_r,
                        -loc.reshape(-1, 3, 1),
                        ).reshape(-1, joint_num, 4, 4)

        # whole transformation point cloud
        repose_mat = torch.matmul(
                        trslt_mat,
                        torch.matmul(rot_mats, back_trslt_mat)
                        )
        # repose_mat = back_trslt_mat            
        
        T = torch.matmul(seg_pc, repose_mat.view(1, joint_num, 16))\
            .view(1, -1, 4, 4)
        pnnocs_pc = lbs_(NOX_pc.unsqueeze(0), T, dtype=NOX_pc.dtype).to(device=NOX_pc.device)

        return pnnocs_pc

    @staticmethod
    def repose_pm_fast(NOX, loc, rot, seg, mask, joint_num, save_pc=False):
        """
        input shape:
            NOX: 3, H, W
            loc: N, 3
            rot: N, 3
            seg: N+2, H, W
            mask: H, W
        """
        
        thresh = 0.7        
        masked = mask.reshape(-1) > thresh

        # t1 = time()
        img_size = NOX.shape
        NOX_pc = NOX.reshape(3, -1)
        NOX_pc *= masked
        NOX_pc = NOX_pc.transpose(0, 1)
        seg_pc = seg.reshape(joint_num+2, -1)
        seg_pc *= masked
        seg_pc = seg_pc.transpose(0, 1)        
        # t3 = time()
        # print(t2-t1, t3-t2)
        num_valid = masked.sum()
        if num_valid == 0:
            # No valid point at all. This will cut off the gradient flow
            return None, None
        to_cat = ()
        # using max_idx to confirm the segmentation
        _, max_idx = seg_pc.max(dim=1, keepdim=True)
                
        seg_flags = range(1, joint_num+2) # 0 is background
        for flag in seg_flags:
            part = (max_idx == flag)
            link = torch.where(part, torch.ones(1, device=NOX.device), torch.zeros(1, device=NOX.device))            
            to_cat = to_cat + (link, )        
        
        seg_pc = torch.cat(to_cat, dim=1)
        loc = loc.unsqueeze(0)
        rot = rot.unsqueeze(0)

        # TODO: following 2 rows would be deleted
        # as here link 2 is the lens with no pose, but we didn't record that
        loc = F.pad(loc, (0, 0, 1, 0), value=0)
        rot = F.pad(rot, (0, 0, 1, 0), value=0)

        # we will add the base joint, it's identical
        joint_num = joint_num + 1

        # rotation
        rodrigues = batch_rodrigues(
                -rot.view(-1, 3),
                dtype=rot.dtype
                ).view([-1, 3, 3])
        I_t = torch.Tensor([0, 0, 0]).to(device=rot.device)\
                    .repeat((joint_num), 1).view(-1, 3, 1)
        rot_mats = transform_mat(
                        rodrigues,
                        I_t,
                        ).reshape(-1, joint_num, 4, 4)

        # translation
        I_r = torch.eye(3).to(device=rot.device)\
                    .repeat(joint_num, 1).view(-1, 3, 3)
        trslt_mat = transform_mat(
                        I_r,
                        loc.reshape(-1, 3, 1),
                        ).reshape(-1, joint_num, 4, 4)
        back_trslt_mat = transform_mat(
                        I_r,
                        -loc.reshape(-1, 3, 1),
                        ).reshape(-1, joint_num, 4, 4)

        # whole transformation point cloud
        repose_mat = torch.matmul(
                        trslt_mat,
                        torch.matmul(rot_mats, back_trslt_mat)
                        )
            
        
        T = torch.matmul(seg_pc, repose_mat.view(1, joint_num, 16))\
            .view(1, -1, 4, 4)
        pnnocs_pc = lbs_(NOX_pc.unsqueeze(0), T, dtype=NOX.dtype).to(device=NOX.device)

        # # re-normalize
        # low_bound = pnnocs_pc.min(axis=1)[0]
        # up_bound = pnnocs_pc.max(axis=1)[0]
        # scale = (up_bound - low_bound).max()

        # if scale != 0:
        #     pnnocs_pc -= low_bound
        #     pnnocs_pc /= scale

        # pnnocs_map = torch.zeros(NOX.size(), device=NOX.device)
        # pnnocs_map[:, masked] = pnnocs_pc.transpose(2, 1)
        # print(pnnocs_pc.shape)
        pnnocs_map = pnnocs_pc.squeeze().reshape(img_size)

        if save_pc:
            pnnocs_pc = pnnocs_map[:, mask > thresh].transpose(0, 1)
        print("here ", pnnocs_pc.shape)
        return pnnocs_pc, pnnocs_map

    def repose_lbs(self, pred_nocs, pred_loc_map, pred_pose_map, pred_weight, conf, pred_mask):
        """
        reposing function for linear blend skining model
        """
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

    def lift(self, output, feature, transform):
        batch_size = output.size(0)
        img_size = (output.size(2), output.size(3))

        feature_dim = feature.shape[1]
        # get masked nocs
        out_mask = output[:, self.nocs_end:self.mask_end, :, :].clone().requires_grad_(True)
        sigmoid = nn.Sigmoid()
        out_mask = sigmoid(out_mask)
        threshold = 0.7

        if self.config.REPOSE:
            pred_nocs = output[:, -3:, :, :].clone().requires_grad_(True)
        else:
            pred_nocs = output[:, :self.nocs_end, :, :].clone().requires_grad_(True)

        # get upsampeld feature
        upsampled_feature = F.interpolate(feature, size=img_size)
        # all_occupancies = []
        
        point_cloud_list = []
        feature_cloud_list = []
        for i in range(batch_size):
            cur_mask = out_mask[i].squeeze()
            masked = cur_mask > threshold
            point_cloud = pred_nocs[i, :, masked]
            
            num_valid = point_cloud.shape[1]
            if num_valid == 0 or torch.isnan(point_cloud).any():
                point_cloud_list.append(None)
                feature_cloud_list.append(None)
                continue

            if self.sample and num_valid > self.max_point:
                pass

            if transform is not None:
                translation = transform['translation'][i].view(3, 1).float()
                point_cloud = point_cloud + translation
                point_cloud = point_cloud * transform['scale'][i]
                point_cloud += 0.5

            feature_cloud = upsampled_feature[i, :, masked]
            
            point_cloud_list.append(point_cloud)
            feature_cloud_list.append(feature_cloud)

            # voxelized_feature = self.discretize(point_cloud, feature_cloud, self.resolution)
            # all_occupancies.append(voxelized_feature)
        
        return point_cloud_list, feature_cloud_list
        # all_occupancies = torch.stack(tuple(all_occupancies))
        # return all_occupancies

    def discretize(self, point_cloud, FeatureCloud, Res):
        # input: N*3 pointcloud
        # output resolution**3 * F
        if point_cloud is None:
            feature_dim = self.SegNet.feature_channels
            return torch.ones(feature_dim, *(self.resolution,)*3).to(device=self.device)        
        feature_dim = FeatureCloud.shape[0]
        point_num = point_cloud.shape[1]

        point_cloud = torch.where(point_cloud < 0, torch.zeros(1, device=point_cloud.device), point_cloud)
        point_cloud = torch.where(point_cloud > 1, torch.ones(1, device=point_cloud.device), point_cloud)

        voxels = torch.floor(point_cloud*Res)
        index = voxels[0, :]*Res**2 + voxels[1, :]*Res + voxels[2, :]
        index = index.unsqueeze(0).to(dtype=torch.long)

        #TODO: replace the mean operation to pointnet
        voxel_feature = torch_scatter.scatter(src=FeatureCloud, index=index)
        # VoxFeature = torch_scatter.segment_coo(src=FeatureCloud,index=Index,reduce='mean')
        pad_size = (0, Res**3 - voxel_feature.size(1))
        voxel_feature = F.pad(voxel_feature, pad_size, 'constant', 0)
        voxel_feature = voxel_feature.view(feature_dim, Res, Res, Res)

        return voxel_feature

    def gen_new_pose(self, input, pose_list):
        """
        Generate new posed shape for given instance
        pose_list: 
        """
        ##############################################################################
        color = inputs['color00']
        output = self.SegNet(color)
        batch_size = output.shape[0]

        pred_nocs = output[:, :self.nocs_end, :, :].clone().requires_grad_(True)
        pred_mask = output[:, self.nocs_end:self.mask_end, :, :].clone().requires_grad_(True)        

        pred_loc = output[:, self.mask_end:self.loc_end, :, :].clone().requires_grad_(True)
        pred_pose = output[:, self.loc_end:self.pose_end, :, :].clone().requires_grad_(True)
        pred_weight = output[:, self.pose_end:self.skin_end, :, :].clone().requires_grad_(True)
        conf = output[:, self.skin_end:self.conf_end, :, :].clone().requires_grad_(True)
        nocs_feature = output[:, self.conf_end:self.ft_end, :, :].clone().requires_grad_(True)


        pnnocs_maps = self.repose_pm_pred(pred_nocs, pred_loc, pred_pose, pred_weight, conf, pred_mask)        
        output = torch.cat((output, pnnocs_maps), dim=1)

        recon = None
        grid_coords = inputs['grid_coords']
        
        transform = {'translation': inputs['translation'],
                    'scale':inputs['scale']}
        
        ######################################################################################
        
        # then: we transform the point cloud into occupancy(along with the features)            
        point_cloud_list, feature_cloud_list = self.lift(output, nocs_feature, transform)
        sv_seg_list = self.get_seg_pc(output)
        # sv_seg = sv_seg_list[0]

        sv_loc_list, sv_rot_list = self.get_pose(output)
        sv_loc = sv_loc_list[0] # this is the location for all the following re-posing
        sv_rot = sv_rot_list[0]
        axis = sv_rot / torch.norm(sv_rot, dim=1).unsqueeze(1)

        loc_list = [sv_rot, ] * len(pose_list)
        axis_angle_list = [axis * p for p in pose_list]

        batch_id = 0
        joint_num = self.joint_num
        posed_pc_list = pose_union(point_cloud_list, sv_seg_list, loc_list, axis_angle_list, joint_num, batch_id)

        occupancy_list = []
        for i in range(batch_size):
            occupancy = self.discretize(posed_pc_list[i], feature_cloud_list[i], self.resolution)
            occupancy_list.append(occupancy)

        occupancies = torch.stack(tuple(occupancy_list))
            # if self.vis == True:
        if 0:
            self.visualize(occupancies)        
        pass
    
    def get_seg_pc(self, sv_output):
        """
        output: list of seg array of shape (N_i, joint_num+2)
        """
        batch_size = sv_output.shape[0]
        thresh = 0.7

        pred_mask = sv_output[:, self.nocs_end:self.mask_end, :, :].clone().requires_grad_(True)
        pred_seg = sv_output[:, self.pose_end:self.skin_end, :, :].clone().requires_grad_(True)
        
        pred_mask = pred_mask.sigmoid().squeeze(1)
        seg_list = []
        for i in range(batch_size):
            cur_mask = pred_mask[i]
            cur_seg = pred_seg[i]
            masked = cur_mask > thresh
            seg_pc = cur_seg[:, masked]
            seg_pc = seg_pc.transpose(0, 1)
            seg_list.append(seg_pc)
        
        return seg_list

    def get_pose(self, sv_output):
        """
        output: list of joint location, list of joint rotation
        """
        batch_size = sv_output.shape[0]
        pred_loc = sv_output[:, self.mask_end:self.loc_end, :, :].clone().requires_grad_(True)
        pred_rot = sv_output[:, self.loc_end:self.pose_end, :, :].clone().requires_grad_(True)
        pred_mask = sv_output[:, self.nocs_end:self.mask_end, :, :].clone().requires_grad_(True)
        conf = sv_output[:, self.skin_end:self.conf_end, :, :].clone().requires_grad_(True)
        pred_mask = pred_mask.sigmoid().squeeze(1)

        pred_loc = self.vote(pred_loc, conf, pred_mask)
        pred_rot = self.vote(pred_rot, conf, pred_mask) # axis-angle representation for the joint
        
        loc_list = []
        rot_list = []

        for i in range(batch_size):
            loc = pred_loc[i].clone().requires_grad_(True)
            rot = pred_rot[i].clone().requires_grad_(True)
            loc_list.append(loc)
            rot_list.append(rot)
        return loc_list, rot_list
    
    @staticmethod
    def pose_union(mv_pn_pc_list, mv_seg_list, mv_loc_list, mv_rot_list, joint_num, batch_id):
        """        
        Output:
            a list of feature enriched occupancy, each item stands for one pose

        Steps:
            1. In the input point cloud and segmentation, we get each point in the PC corresponds to
                a seg flag in the segmentation, so if we just concate the point cloud and segmentation
                individually, the new point cloud is still segmented
            2. Then we call the `repose_core` function, bring the point cloud to a new pose. 
                We do this for EACH pose
            3. we output the list of merged and posed point clouds. 
        """
        inst_pn_pc_list = []
        inst_seg_list = []
        inst_loc_list = []
        inst_rot_list = []
        # inst_feature_list = []        
        valid_view_num = len(mv_pn_pc_list)

        for view in range(valid_view_num):
            cur_pc = mv_pn_pc_list[view][batch_id]            
            cur_loc = mv_loc_list[view][batch_id]
            cur_rot = mv_rot_list[view][batch_id]
            cur_seg = mv_seg_list[view][batch_id]
            # cur_feature = mv_feature_list[view][batch_id]
            if cur_pc is not None:
                # input with shape (3, N)
                inst_pn_pc_list.append(cur_pc.transpose(0, 1))
                inst_seg_list.append(cur_seg)
                inst_loc_list.append(cur_loc)
                inst_rot_list.append(cur_rot)
                # inst_feature_list.append(cur_feature)
        if len(inst_pn_pc_list) == 0:
            return None
            # inst_pn_pc = None
            # inst_feature = None
        else:
            inst_pn_pc = torch.cat(tuple(inst_pn_pc_list), dim=0)
            inst_seg = torch.cat(tuple(inst_seg_list), dim=0)
            # inst_feature = torch.cat(tuple(inst_feature_list), dim=1)        
        
        posed_pc_list = []
        for i in range(len(inst_loc_list)):
            loc = inst_loc_list[i]
            rot = inst_rot_list[i]
            # write("/workspace/debug_0_loc.xyz",loc)
            # write("/workspace/debug_0_rot.xyz",rot)
            rot = -rot

            posed_pc = LNRNet.repose_pc(inst_pn_pc, inst_seg, loc, rot, joint_num=joint_num)
            posed_pc = posed_pc[0].transpose(1, 0)
            posed_pc_list.append(posed_pc)
        
        return posed_pc_list


    def visualize(self, FeatureVoxel):
        feature_sample = FeatureVoxel[0]
        voxel = (feature_sample != 0).sum(dim=0).to(dtype=torch.bool)
        
        self.debug_dir = os.path.join(self.config.OUTPUT_DIR, self.config.EXPT_NAME, "debug")
        if not os.path.exists(self.debug_dir):
            os.mkdir(self.debug_dir)
        off_path = os.path.join(self.debug_dir, "mid.off")
        # gt_path = "/workspace/dev_nrnocs/debug/gt.off"

        VoxelGrid(voxel.cpu().detach(), (0, 0, 0), 1).to_mesh().export(off_path)

        # mesh_path = "/workspace/Data/IF_PN_Aug13/train/0000/frame_00000000_isosurf_scaled.off"
        # import trimesh
        # mesh = trimesh.load(mesh_path)
        # vox = VoxelGrid.from_mesh(mesh, self.resolution, loc=[0, 0, 0], scale=1)
        # vox.to_mesh().export(gt_path)
        exit()

    