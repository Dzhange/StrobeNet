import torch.nn as nn
import torchvision.models as models
import os, sys
FileDirPath = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(FileDirPath, '..'))
sys.path.append(os.path.join(FileDirPath, '../..'))
from models.networks.modules import *
from models.networks.TripleHeadSegNet import THSegNet

class MVTHSegNet(THSegNet):
    """
    Multi-view triple headed segnet
    """
    def __init__(self, nocs_channels=4, pose_channels=48+48+16+16,
                 input_channels=3,
                 feature_channels=64,
                 pretrained=True, withSkipConnections=True, bn=True):

        super().__init__(nocs_channels=nocs_channels, pose_channels=pose_channels,
                        input_channels=input_channels,
                        feature_channels=feature_channels,
                        pretrained=pretrained, withSkipConnections=withSkipConnections, bn=bn)
    
    def forward(self, inputs):
        """
        From jiahui's implementation in https://github.com/JiahuiLei/Pix2Surf
        inputs: list of rgb images
        output: prediction's cated in order of :
            NOCS, POSE, Features
        """
        return_code=False
        share_feature=False
        
        l1, l2, l3, l4, l5 = [], [], [], [], []
        l5_feature = []
        for rgb in inputs:
            down1, indices_1, unpool_shape1, FM1 = self.down1(rgb)
            down2, indices_2, unpool_shape2, FM2 = self.down2(down1)
            down3, indices_3, unpool_shape3, FM3 = self.down3(down2)
            down4, indices_4, unpool_shape4, FM4 = self.down4(down3)
            down5, indices_5, unpool_shape5, FM5 = self.down5(down4)
            
            l1.append([indices_1, unpool_shape1, FM1])
            l2.append([indices_2, unpool_shape2, FM2])
            l3.append([indices_3, unpool_shape3, FM3])
            l4.append([indices_4, unpool_shape4, FM4])
            l5.append([indices_5, unpool_shape5, FM5])
            l5_feature.append(down5.unsqueeze(0))

        # here cated features from the bottle neck
        # seget shares fetures among different views
        # TODO: maybe we can use this to disentangle shape and pose
        if share_feature:
            max_pooled_feature = torch.max(torch.cat(l5_feature, 0), dim=0).values
            f_dim = max_pooled_feature.shape[1]

        pred_list = []
        for i in range(len(inputs)):
            if share_feature:
                down5 = torch.cat((max_pooled_feature[:, :f_dim // 2, :, :],
                               l5_feature[i].squeeze(0)[:, f_dim // 2:, :, :]), dim=1)

            up5 = self.up5(down5, *l5[i])
            up4 = self.up4(up5, *l4[i])
            up3 = self.up3(up4, *l3[i])
            up2 = self.up2(up3, *l2[i])

            nocs_output = self.nocs_head(up2, *l1[i])
            pose_output = self.pose_head(up2, *l1[i])
            feature_output = self.feature_head(up2, *l1[i])
            output = torch.cat((nocs_output, pose_output, feature_output), dim=1)
            
            pred_list.append(output)

        feature_list = [item.squeeze(0) for item in l5_feature]

        if return_code:
            return pred_list, feature_list
        else:
            return pred_list


    
if __name__ == '__main__':
    import torch

    net = MVTHSegNet(withSkipConnections=True).cuda()
    x = torch.rand(2, 3, 640, 480).cuda()
    y = net(x)
    print(y.shape)
