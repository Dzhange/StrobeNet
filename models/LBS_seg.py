import os
import glob
import torch
import torch.nn as nn
from models.LBS import ModelLBSNOCS
from models.networks.SegNet import SegNet as old_SegNet
from models.networks.say4n_SegNet import SegNet as new_SegNet
from models.loss import LBSLoss
from utils.DataUtils import *

class LBS_seg(ModelLBSNOCS):

    def __init__(self, config):
        super().__init__(config)

    

