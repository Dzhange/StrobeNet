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
        
        self.resolution = 128
        self.init_grids(self.resolution)
        self.UpdateSegNet = config.UPDATE_SEG
        self.use_pretrained = False
        self.init_hp()
        
        # Freeze the SegNet part due to the bug
        # if self.FreezeSegNet:
        for param in self.SegNet.parameters():
            param.requires_grad = self.UpdateSegNet
    
    def init_network(self):
        self.SegNet = THSegNet(pose_channels=self.joint_num*(3+3+1+1)+2, bn=self.config.BN)
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
        self.ft_end = self.conf_end + 64

    def init_grids(self, resolution):
        if self.transform:
            bb_min = -0.5
            bb_max = 0.5
        else:
            bb_min = 0
            bb_max = 1

        self.GridPoints = iw.create_grid_points_from_bounds(bb_min, bb_max, resolution)
        grid_points = iw.create_grid_points_from_bounds(bb_min, bb_max, resolution)
        grid_points[:, 0], grid_points[:, 2] = grid_points[:, 2], grid_points[:, 0].copy()        
        a = bb_max + bb_min
        b = bb_max - bb_min
        grid_coords = 2 * grid_points - a
        grid_coords = grid_coords / b
        grid_coords = torch.from_numpy(grid_coords).to(self.device, dtype=torch.float)
        grid_coords = torch.reshape(grid_coords, (1, len(grid_points), 3)).to(self.device)
        self.grid_coords = grid_coords
        # return grid_coords
    
    def forward(self, inputs):
        color = inputs['color00']

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
        pred_joint_score = self.sigmoid(pred_joint_score) * out_mask.unsqueeze(1)
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
        threshold = 0.75
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
        sigmoid = nn.Sigmoid()
        pred_mask = sigmoid(pred_mask).squeeze(1)        
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
            if torch.isnan(nocs).any() or torch.isnan(seg).any() or torch.isnan(mask).any():
                all_pnnocs.append(torch.zeros(pred_nocs[0].size()).to(device=pred_nocs.device))
            else:
                pnnocs, pnnocs_map = self.repose_pm_core(nocs, loc, rot, seg, mask, self.joint_num)
                if pnnocs is None:
                    all_pnnocs.append(torch.zeros(pred_nocs[0].size()).to(device=pred_nocs.device))
                else:
                    all_pnnocs.append(pnnocs_map)
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
        thresh = 0.75
        masked = mask > thresh
        NOX_pc = NOX[:, masked].transpose(0, 1)
        seg_pc = seg[:, masked].transpose(0, 1)

        num_valid = NOX_pc.shape[0]
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

        pnnocs_map = torch.zeros(NOX.size(), device=NOX.device)
        pnnocs_map[:, masked] = pnnocs_pc.transpose(2, 1)

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
        threshold = 0.75

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
        # output: 128**3 * F
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
        # vox = VoxelGrid.from_mesh(mesh, 128, loc=[0, 0, 0], scale=1)
        # vox.to_mesh().export(gt_path)
        exit()