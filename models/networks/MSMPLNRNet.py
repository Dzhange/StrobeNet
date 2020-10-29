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

class MSMPLNRNet(LNRNet):

    def __init__(self, config):
        super().__init__(config)

    def forward(self, inputs):

        img_list = inputs['color00']
        batch_size = self.config.BATCHSIZE

        # DEBUG: try old network
        
        if isinstance(self.SegNet, MVTHSegNet):
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
        if isinstance(inputs['grid_coords'], list):
            grid_coords = inputs['grid_coords'][0] # grid coords should be identical among all views
        else:
            grid_coords = inputs['grid_coords'] # some times(in validation time, we feed it with one tensor)
        if not self.config.STAGE_ONE:
            if self.aggr_scatter:
                occupancy_list = []
                for i in range(batch_size):
                    # we aggregate for each item in the batch
                    instance_pc, instance_feature = self.collect_pc(mv_pc_list, mv_feature_list)
                    occupancy = self.discretize(instance_pc, instance_feature, self.resolution)
                    occupancy_list.append(occupancy)
                occupancies = torch.stack(tuple(occupancy_list))
            else:
                mv_occupancy = torch.stack(tuple(b_mv_occupancy_list), dim=1)
                mv_occupancy = self.avgpool_grids(mv_occupancy)

            # self.visualize(occupancies)
            # and then feed into IF-Net. The ground truth shouled be used in the back projection
            recon = self.IFNet(grid_coords, occupancies)
        
        # return output_list, recon
        if isinstance(self.SegNet, MVTHSegNet):
            return output_list, recon
        else:
            return sv_output, recon

    @staticmethod
    def pose_union(sv_outpuit_list):
        """
        Input:
            a list of ``reposed`` SegNet output, each one comes from one view            
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
        pass




        # thresh = 0.75
        # masked = mask > thresh
        # # t1 = time()        
        # NOX_pc = NOX[:, masked]
        # seg_pc = seg[:, masked]
        # # t2 = time()
        # NOX_pc = NOX_pc.transpose(0, 1)
        # seg_pc = seg_pc.transpose(0, 1)
        



        


        


                
