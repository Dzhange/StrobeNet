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
from utils.DataUtils import *

class MVMPLNRNet(MLNRNet):

    def __init__(self, config, device):
        super().__init__(config, device=device)
        # self.aggr_scatter = config.AGGR_SCATTER
        # self.view_num = config.VIEW_NUM
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
                # pc in sv_pn_pc_list are transformed and added offset for discrestization
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
        pn_recon = None    
        posed_recon_list = []

        if not self.config.STAGE_ONE:
            pn_occupancy_list = [] # list for multi instance
            batch_posed_occupancy_list = [] # value is list of multi view occupancy in each instance
            for i in range(batch_size):
                # aggregatre pose normalized view in each batch
                inst_pn_pc, inst_feature = self.collect_pc(mv_pn_pc_list, mv_feature_list, batch_id=i)
                pn_occupancy = self.discretize(inst_pn_pc, inst_feature, self.resolution)
                pn_occupancy_list.append(pn_occupancy)
                
                # repose uniton point cloud into individual pose 
                inst_posed_pc_list = self.pose_union(mv_pn_pc_list, mv_seg_list, mv_loc_list, mv_rot_list,\
                                                         joint_num=self.joint_num, batch_id=i)

                posed_occupancy_list = [] # value is occupancy of multi view of 1 instance
                # print((inst_posed_pc_list[0] - inst_posed_pc_list[1]).sum())
                for posed_pc in inst_posed_pc_list:
                    # print(posed_pc.shape)
                    # write_off('/workspace/debug_0.xyz', posed_pc.transpose(0, 1).cpu().detach().numpy())
                    # exit()
                    posed_occupancy = self.discretize(posed_pc, inst_feature, self.resolution)
                    posed_occupancy_list.append(posed_occupancy)
                batch_posed_occupancy_list.append(posed_occupancy_list)

            # self.visualize(occupancies)
            # and then feed into IF-Net. The ground truth shouled be used in the back projection
            pn_occupancies = torch.stack(tuple(pn_occupancy_list))
            pn_grid_coords = inputs['cano_grid_coords']
            pn_recon = self.IFNet(pn_grid_coords, pn_occupancies)
                                    
            for view_id in range(self.view_num):
                view_occ = []
                # accumulate across batch
                for b_id in range(batch_size):
                    view_occ.append(batch_posed_occupancy_list[b_id][view_id])
                one_pose_occ = torch.stack(tuple(view_occ))
                # self.visualize(one_pose_occ)
                grid_coords = inputs['grid_coords'][view_id]
                posed_recon = self.IFNet(grid_coords, one_pose_occ)
                posed_recon_list.append(posed_recon)
                
        return output_list, pn_recon, posed_recon_list
        




