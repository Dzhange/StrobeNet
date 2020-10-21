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
    
    def init_net(self, device):
        print("MLNR init")
        config = self.config
        self.net = MLNRNet(config, device=device)
        self.optimizer = torch.optim.Adam(params=self.net.parameters(), lr=self.lr,
                                          betas=(self.config.ADAM_BETA1, self.config.ADAM_BETA2))        
        if config.NRNET_PRETRAIN:
            if config.NRNET_PRETRAIN_PATH.endswith('tar'):
                self.LoadSegNetFromTar(device, config.NRNET_PRETRAIN_PATH)
            else:
                pretrained_dir = config.NRNET_PRETRAIN_PATH
                self.LoadSegNetCheckpoint(device, pretrained_dir)

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
        no_compute_item = ['mesh', 'iso_mesh']
        input_items = ['color00', 'grid_coords', 'translation', 'scale']
        target_items = ['nox00', 'pnnocs00', 'joint_map', 'linkseg', 'occupancies', 'pose']
        
        inputs = {}
        targets = {}

        for k in data:        
            if k in no_compute_item:                
                targets[k] = data[k][0] # data['mesh] = [('p1','p2')]
            else:
                ondevice_data = [item.to(device=device) for item in data[k]]
                if k in input_items:
                    inputs[k] = ondevice_data
                if k in target_items:
                    targets[k] = ondevice_data
        return inputs, targets




