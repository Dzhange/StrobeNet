# Borrowed from https://github.com/meetshah1995/pytorch-semseg
import torch.nn as nn
import os, sys

import torchvision.models as models

from models.networks.SegNet import SegNet
from models.networks.modules import segnetDown2, segnetDown3, segnetUp2, segnetUp3

import numpy as np
import torch 

# SegNet version specified for NRNet
class SegNetNR(SegNet):
    def __init__(self, output_channels=8, input_channels=3,  pretrained=True, withSkipConnections=True):
        super().__init__(output_channels=output_channels, pretrained=pretrained, withSkipConnections=withSkipConnections)

    def forward(self, inputs):
        down1, indices_1, unpool_shape1, FM1 = self.down1(inputs)
        down2, indices_2, unpool_shape2, FM2 = self.down2(down1)
        down3, indices_3, unpool_shape3, FM3 = self.down3(down2)
        down4, indices_4, unpool_shape4, FM4 = self.down4(down3)
        down5, indices_5, unpool_shape5, FM5 = self.down5(down4)

        up5 = self.up5(down5, indices_5, unpool_shape5, SkipFeatureMap=FM5)
        up4 = self.up4(up5, indices_4, unpool_shape4, SkipFeatureMap=FM4)
        up3 = self.up3(up4, indices_3, unpool_shape3, SkipFeatureMap=FM3)
        up2 = self.up2(up3, indices_2, unpool_shape2, SkipFeatureMap=FM2)
        up1 = self.up1(up2, indices_1, unpool_shape1, SkipFeatureMap=FM1)

        # return up1, down5
        return up1, up3    
   
