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
from models.networks.IFNet import SVR
from models.networks.ShallowIFNet import ShallowSVR
from models.networks.LNRNet import LNRNet
from models.networks.MLNRNet import MLNRNet
from time import time
import utils.tools.implicit_waterproofing as iw
from utils.lbs import *
from utils.tools.voxels import VoxelGrid
from utils.tools.pc2voxel import voxelize as pc2vox
from utils.DataUtils import DL2LD, LD2DL

class MVMPLNRNet(LNRNet):

    def __init__(self, config):
        super().__init__(config)
        assert self.aggr_scatter

    def forward(self, inputs):

        img_list = inputs['color00']
        batch_size = self.config.BATCHSIZE

        # DEBUG: try old network
        
        if isinstance(self.SegNet, MVTHSegNet):
            pred_list = self.SegNet(img_list)
        else:
            sv_output = self.SegNet(img_list[0])
        mv_pn_pc_list = []
        mv_feature_list = []
        mv_seg_list = []
        mv_loc_list = []
        mv_rot_list = []

        b_mv_occupancy_list = [] # occupancy stored in it are batch wise

        if self.transform:
            transform = {'translation': inputs['translation'],
                     'scale':inputs['scale']}
            transform_list = DL2LD(transform)
        else:
            transform_list = None

        ##### lifting, iterate through all views #####
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
                sv_pn_pc_list, sv_feature_list = self.lift(sv_output, nocs_feature, transform_list[i])
                sv_seg_list = self.get_seg_pc(sv_output)
                sv_loc_list, sv_rot_list = self.get_pose(sv_output)
                # this list is batch wise
                if sv_pn_pc_list is not None and sv_feature_list is not None:
                    mv_pn_pc_list.append(sv_pn_pc_list)
                    mv_feature_list.append(sv_feature_list)
                    mv_seg_list.append(sv_seg_list)
                    mv_loc_list.append(sv_loc_list)
                    mv_rot_list.append(sv_rot_list)

        ##### aggregation, iterate through all instance #####
        # recon = None        
        if not self.config.STAGE_ONE:
            pn_occupancy_list = []
            for i in range(batch_size):
                # aggregatre pose normalized view in each batch
                inst_pn_pc, inst_feature = self.collect_pc(mv_pn_pc_list, mv_feature_list, batch_id=i)
                pn_occupancy = self.discretize(inst_pn_pc, inst_feature, self.resolution)
                pn_occupancy_list.append(pn_occupancy)
                
                
                # repose uniton point cloud into individual pose 
                inst_posed_pc_list = self.pose_union(mv_pn_pc_list, mv_seg_list, mv_loc_list, mv_rot_list,\
                                                         joint_num=self.joint_num, batch_id=i)

                posed_occupancy_list = []
                for posed_pc in inst_posed_pc_list:
                    posed_occupancy = self.discretize(inst_pn_pc, inst_feature, self.resolution)
                    posed_occupancy_list.append(posed_occupancy)
                        
            # self.visualize(occupancies)
            # and then feed into IF-Net. The ground truth shouled be used in the back projection
            pn_occupancies = torch.stack(tuple(pn_occupancy_list))
            pn_grid_coords = inputs['cano_grid_coords'][0]
            pn_recon = self.IFNet(pn_grid_coords, pn_occupancies)
            
            posed_recon_list = []            
            for view_id, posed_occupancy in enumerate(posed_occupancy_list):
                grid_coords = inputs['cano_grid_coords'][view_id]
                posed_recon = self.IFNet(grid_coords, pn_occupancies)
                posed_recon_list.append(posed_recon)
                
        return output_list, pn_recon, posed_recon_list
        

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
                inst_pn_pc_list.append(cur_pc)
                inst_seg_list.append(cur_seg)
                inst_loc_list.append(cur_loc)
                inst_rot_list.append(cur_rot)
                # inst_feature_list.append(cur_feature)
        if len(inst_pn_pc_list) == 0:
            return None
            # inst_pn_pc = None
            # inst_feature = None
        else:
            inst_pn_pc = torch.cat(tuple(inst_pn_pc_list), dim=1)
            inst_seg = torch.cat(tuple(inst_seg_list), dim=1)
            # inst_feature = torch.cat(tuple(inst_feature_list), dim=1)        
        
        posed_pc_list = []
        for i in range(len(inst_loc_list)):
            loc = inst_loc_list[i]
            rot = inst_loc_list[i]
            posed_pc = LNRNet.repose_pc(inst_pn_pc, inst_seg, loc, rot, joint_num=joint_num)
            posed_pc_list.append(posed_pc)
        
        return posed_pc_list


        


        


                
