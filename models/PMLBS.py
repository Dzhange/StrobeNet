import os
import glob
import torch
import torch.nn as nn
from models.SegLBS import ModelSegLBS
from models.networks.MultiHeadSegNet import MHSegNet
from utils.DataUtils import *

class PMLBS(ModelSegLBS):
    """
    The network trains for partial mobility objects
    """
    def __init__(self, config):
        super().__init__(config)
        self.bone_num = 3 # eyeglasses
        self.net = MHSegNet(bn=False, pose_channels=2*(3+3+1+1))

    @staticmethod
    def preprocess(data, device):
        """
        put data onto the right device
        data input:
            0: color image
            1: loaded maps for supervision:
                1) segmap
                2) joint map
            2: Pose in np format
            3: occupancies
            4: GT mesh path
        """
        data_to_device = []
        for item in data:
            tuple_or_tensor = item
            tuple_or_tensor_td = item
            if isinstance(tuple_or_tensor_td, list):
                for ctr in range(len(tuple_or_tensor)):
                    if isinstance(tuple_or_tensor[ctr], torch.Tensor):
                        tuple_or_tensor_td[ctr] = tuple_or_tensor[ctr].to(device)
                    else:
                        tuple_or_tensor_td[ctr] = tuple_or_tensor[ctr]
                data_to_device.append(tuple_or_tensor_td)
            elif isinstance(tuple_or_tensor_td, (dict)):
                dict_td = {}
                keys = item.keys()
                for key in keys:
                    if isinstance(item[key], torch.Tensor):
                        dict_td[key] = item[key].to(device)
                data_to_device.append(dict_td)
            elif isinstance(tuple_or_tensor, torch.Tensor):
                tuple_or_tensor_td = tuple_or_tensor.to(device)
                data_to_device.append(tuple_or_tensor_td)
            else:
                # for gt mesh
                continue
        inputs = {}
        inputs['RGB'] = data_to_device[0]
        inputs['grid_coords'] = data_to_device[3]['grid_coords']
        inputs['translation'] = data_to_device[3]['translation']
        inputs['scale'] = data_to_device[3]['scale']

        targets = {}
        targets['maps'] = data_to_device[1]
        targets['pose'] = data_to_device[2]
        targets['occupancies'] = data_to_device[3]['occupancies']
        targets['mesh'] = data[4]
        return inputs, targets