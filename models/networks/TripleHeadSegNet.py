import torch.nn as nn
import torchvision.models as models
import os, sys
FileDirPath = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(FileDirPath, '..'))
sys.path.append(os.path.join(FileDirPath, '../..'))
from models.networks.modules import *

class THSegNet(nn.Module):
    def __init__(self, nocs_channels=4, pose_channels=48+48+16+16,
                 input_channels=3,
                 feature_channels=64,
                 pretrained=True, withSkipConnections=True, bn=True):

        super().__init__()
        self.in_channels = input_channels    
        self.withSkipConnections = withSkipConnections
        self.down1 = segnetDown2(self.in_channels, 64, withFeatureMap=self.withSkipConnections, bn=bn)
        self.down2 = segnetDown2(64, 128, withFeatureMap=self.withSkipConnections, bn=bn)
        self.down3 = segnetDown3(128, 256, withFeatureMap=self.withSkipConnections, bn=bn)
        self.down4 = segnetDown3(256, 512, withFeatureMap=self.withSkipConnections, bn=bn)
        self.down5 = segnetDown3(512, 512, withFeatureMap=self.withSkipConnections, bn=bn)

        self.up5 = segnetUp3(512, 512, withSkipConnections=self.withSkipConnections, bn=bn)
        self.up4 = segnetUp3(512, 256, withSkipConnections=self.withSkipConnections, bn=bn)
        self.up3 = segnetUp3(256, 128, withSkipConnections=self.withSkipConnections, bn=bn)
        self.up2 = segnetUp2(128, 64, withSkipConnections=self.withSkipConnections, bn=bn)

        self.nocs_head = segnetUp2(64, nocs_channels, last_layer=True, withSkipConnections=self.withSkipConnections, bn=bn)
        self.pose_head = segnetUp2(64, pose_channels, last_layer=True, withSkipConnections=self.withSkipConnections, bn=bn)        
        self.feature_head = segnetUp2(64, feature_channels, last_layer=True, withSkipConnections=self.withSkipConnections, bn=bn)

        if pretrained:
            vgg16 = models.vgg16(pretrained=True)
            Arch = 'SegNet'
            if self.withSkipConnections:
                Arch = 'SegNetSkip'
            print('[ INFO ]: Using pre-trained weights from VGG16 with {}.'.format(Arch))
            self.init_vgg16_params(vgg16)

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

        nocs_output = self.nocs_head(up2, indices_1, unpool_shape1, SkipFeatureMap=FM1)
        pose_output = self.pose_head(up2, indices_1, unpool_shape1, SkipFeatureMap=FM1)
        feature_output = self.feature_head(up2, indices_1, unpool_shape1, SkipFeatureMap=FM1)

        output = torch.cat((nocs_output, pose_output, feature_output), dim=1)
        return output

    def init_vgg16_params(self, vgg16):
        blocks = [self.down1, self.down2, self.down3, self.down4, self.down5]

        features = list(vgg16.features.children())

        vgg_layers = []
        for _layer in features:
            if isinstance(_layer, nn.Conv2d):
                vgg_layers.append(_layer)

        merged_layers = []
        for idx, conv_block in enumerate(blocks):
            if idx < 2:
                units = [conv_block.conv1.cbr_unit, conv_block.conv2.cbr_unit]
            else:
                units = [
                    conv_block.conv1.cbr_unit,
                    conv_block.conv2.cbr_unit,
                    conv_block.conv3.cbr_unit,
                ]
            for _unit in units:
                for _layer in _unit:
                    if isinstance(_layer, nn.Conv2d):
                        merged_layers.append(_layer)

        assert len(vgg_layers) == len(merged_layers)

        for l1, l2 in zip(vgg_layers, merged_layers):
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                assert l1.weight.size() == l2.weight.size()
                assert l1.bias.size() == l2.bias.size()
                l2.weight.data = l1.weight.data
                l2.bias.data = l1.bias.data


if __name__ == '__main__':
    import torch

    net = MHSegNet(withSkipConnections=True).cuda()
    x = torch.rand(2, 3, 640, 480).cuda()
    y = net(x)
    print(y.shape)
