"""
This is the multi-view version of LNRNOCS
We use Multi-View SegNet for the first stage
and then use the network to predict it back.
"""
import os, sys
import shutil
import glob
import torch
import torch.nn as nn
from models.networks.MLNRNet import MLNRNet
from models.LNR import ModelLNRNET
from models.SegLBS import ModelSegLBS
from utils.DataUtils import *
from utils.lbs import *
import trimesh, mcubes

class ModelMLNRNet(ModelLNRNET):

    def __init__(self, config):
        super().__init__(config)
        self.view_num = config.VIEW_NUM
    
    def preprocess(self, data, device):
        """
        put data onto the right device
        'color00', 'nox00', 'linkseg', 'pnnocs00'
        data input(is a dict):
            keys include:
            # for segnet
            1. color00
            2. linkseg
            3. nox00
            4. pnnocs00
            5. joint map
            6. pose

            # for if-net
            7 grid_coords
            8. occupancies
            9. translation
            10. scale
        """
        no_compute_item = ['mesh']
        input_items = ['color00', 'grid_coords', 'translation', 'scale']
        target_items = ['nox00', 'pnnocs00', 'joint_map', 'linkseg', 'occupancies']
        
        inputs = {}
        targets = {}

        for k in data:
            if k in no_compute_item:
                target_items[k] = data[k]
            else:
                if k in input_items:
                    inputs[k] = data[k].to(device=device)
                if k in target_items:
                    targets[k] = data[k].to(device=device)                
        return inputs, targets



