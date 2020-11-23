# Copied from https://github.com/jchibane/if-net

import torch
import torch.nn as nn
import torch.nn.functional as F

# 1D convolution is used for the decoder. It acts as a standard FC, but allows to use a batch of point samples features,
# additionally to the batch over the input objects.
# The dimensions are used as follows:
# batch_size (N) = #3D objects , channels = features, signal_lengt (L) (convolution dimension) = #point samples
# kernel_size = 1 i.e. every convolution is done over only all features of one point sample, this makes it a FC.

# 3D Single View Reconsturction (for 256**3 input voxelization) --------------------------------------
# ----------------------------------------------------------------------------------------------------
class SVR(nn.Module):

    def __init__(self, config, device, hidden_dim=256):
        super(SVR, self).__init__()

        self.config = config
        if self.config.USE_FEATURE:
            input_dim = config.FEATURE_CHANNELS
        else:
            input_dim = 1 # no feature, only occupancy

        self.conv_in = nn.Conv3d(input_dim, self.config.IF_IN_DIM, 3, padding=1, padding_mode='replicate')  # out: 256 ->m.p. 128
        self.conv_0 = nn.Conv3d(self.config.IF_IN_DIM, 32, 3, padding=1, padding_mode='replicate')  # out: 128
        self.conv_0_1 = nn.Conv3d(32, 32, 3, padding=1, padding_mode='replicate')  # out: 128 ->m.p. 64
        self.conv_1 = nn.Conv3d(32, 64, 3, padding=1, padding_mode='replicate')  # out: 64
        self.conv_1_1 = nn.Conv3d(64, 64, 3, padding=1, padding_mode='replicate')  # out: 64 -> mp 32
        self.conv_2 = nn.Conv3d(64, 128, 3, padding=1, padding_mode='replicate')  # out: 32
        self.conv_2_1 = nn.Conv3d(128, 128, 3, padding=1, padding_mode='replicate')  # out: 32 -> mp 16
        self.conv_3 = nn.Conv3d(128, 128, 3, padding=1, padding_mode='replicate')  # out: 16
        self.conv_3_1 = nn.Conv3d(128, 128, 3, padding=1, padding_mode='replicate')  # out: 16 -> mp 8
        self.conv_4 = nn.Conv3d(128, 128, 3, padding=1, padding_mode='replicate')  # out: 8
        self.conv_4_1 = nn.Conv3d(128, 128, 3, padding=1, padding_mode='replicate')  # out: 8
        
        feature_size = (input_dim + self.config.IF_IN_DIM + 32 + 64 + 128 + 128 + 128) * 7 + 3
        
        self.use_global_feature = config.GLOBAL_FEATURE
        if self.use_global_feature:
            feature_size += 1024
        if self.config.GLOBAL_ONLY:
            feature_size = 1024

        self.fc_0 = nn.Conv1d(feature_size, hidden_dim * 2, 1)
        self.fc_1 = nn.Conv1d(hidden_dim *2, hidden_dim, 1)
        self.fc_2 = nn.Conv1d(hidden_dim , hidden_dim, 1)
        self.fc_out = nn.Conv1d(hidden_dim, 1, 1)
        self.actvn = nn.ReLU()

        self.maxpool = nn.MaxPool3d(2)

        ## try not using bn
        self.use_bn = self.config.IF_BN
        if self.use_bn:
            print("we should not be here")
            track_stats = True
            self.conv_in_bn = nn.BatchNorm3d(self.config.IF_IN_DIM, track_running_stats=track_stats)
            self.conv0_1_bn = nn.BatchNorm3d(32, track_running_stats=track_stats)
            self.conv1_1_bn = nn.BatchNorm3d(64, track_running_stats=track_stats)
            self.conv2_1_bn = nn.BatchNorm3d(128, track_running_stats=track_stats)
            self.conv3_1_bn = nn.BatchNorm3d(128, track_running_stats=track_stats)
            self.conv4_1_bn = nn.BatchNorm3d(128, track_running_stats=track_stats)

        displacment = 0.0722
        displacments = []
        displacments.append([0, 0, 0])
        for x in range(3):
            for y in [-1, 1]:
                input = [0, 0, 0]
                input[x] = y * displacment
                displacments.append(input)

        self.displacments = torch.Tensor(displacments).to(device)

    def forward(self, p, x, global_feature=None):
        # x = x.unsqueeze(1)

        p_features = p.transpose(1, -1)
        p = p.unsqueeze(1).unsqueeze(1)
        p = torch.cat([p + d for d in self.displacments], dim=2)
        feature_0 = F.grid_sample(x, p, padding_mode='border')

        net = self.actvn(self.conv_in(x))
        if self.use_bn:
            net = self.conv_in_bn(net)
        feature_1 = F.grid_sample(net, p, padding_mode='border')
        net = self.maxpool(net) #out 128

        net = self.actvn(self.conv_0(net))
        net = self.actvn(self.conv_0_1(net))
        if self.use_bn:
            net = self.conv0_1_bn(net)
        feature_2 = F.grid_sample(net, p, padding_mode='border')
        net = self.maxpool(net) #out 64

        net = self.actvn(self.conv_1(net))
        net = self.actvn(self.conv_1_1(net))
        if self.use_bn:
            net = self.conv1_1_bn(net)
        feature_3 = F.grid_sample(net, p, padding_mode='border')
        net = self.maxpool(net)

        net = self.actvn(self.conv_2(net))
        net = self.actvn(self.conv_2_1(net))
        if self.use_bn:
            net = self.conv2_1_bn(net)
        feature_4 = F.grid_sample(net, p, padding_mode='border')
        net = self.maxpool(net)

        net = self.actvn(self.conv_3(net))
        net = self.actvn(self.conv_3_1(net))
        if self.use_bn:
            net = self.conv3_1_bn(net)
        feature_5 = F.grid_sample(net, p, padding_mode='border')
        net = self.maxpool(net)

        net = self.actvn(self.conv_4(net))
        net = self.actvn(self.conv_4_1(net))
        if self.use_bn:
            net = self.conv4_1_bn(net)
        feature_6 = F.grid_sample(net, p, padding_mode='border')

        # here every channel corresponse to one feature.

        features = torch.cat((feature_0, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6),
                             dim=1)  # (B, features, 1,7,sample_num)

        


        shape = features.shape
        features = torch.reshape(features,
                                 (shape[0], shape[1] * shape[3], shape[4]))  # (B, featues_per_sample, samples_num)
        features = torch.cat((features, p_features), dim=1)  # (B, featue_size, samples_num)
                
        if self.config.GLOBAL_ONLY:
            features = global_feature.view(-1, 1).repeat(1, 1, features.shape[2])
        elif self.use_global_feature:
            global_feature = global_feature.view(-1, 1).repeat(1, 1, features.shape[2])
            # print(global_feature.shape, features.shape)
            features = torch.cat((features, global_feature), dim=1)
        

        # print(features.shape)
        net = self.actvn(self.fc_0(features))
        # print(net.shape)
        net = self.actvn(self.fc_1(net))
        # print(net.shape)
        net = self.actvn(self.fc_2(net))
        # print(net.shape)
        net = self.fc_out(net)
        out = net.squeeze(1)

        return out


class SuperRes(nn.Module):

    def __init__(self, config, device, hidden_dim=256):
        super(SuperRes, self).__init__()

        self.config = config
        input_dim = config.FEATURE_CHANNELS
        # accepts 128**3 res input
        # self.conv_in = nn.Conv3d(1, 16, 3, padding=1)  # out: 128
        self.conv_in = nn.Conv3d(input_dim, self.config.IF_IN_DIM, 3, padding=1, padding_mode='replicate')  # out: 256 ->m.p. 128
        # self.conv_0 = nn.Conv3d(16, 32, 3, padding=1)  # out: 64
        self.conv_0 = nn.Conv3d(self.config.IF_IN_DIM, 32, 3, padding=1)  # out: 64
        self.conv_0_1 = nn.Conv3d(32, 32, 3, padding=1)  # out: 64
        self.conv_1 = nn.Conv3d(32, 64, 3, padding=1)  # out: 32
        self.conv_1_1 = nn.Conv3d(64, 64, 3, padding=1)  # out: 32
        self.conv_2 = nn.Conv3d(64, 128, 3, padding=1)  # out: 16
        self.conv_2_1 = nn.Conv3d(128, 128, 3, padding=1)  # out: 16
        self.conv_3 = nn.Conv3d(128, 128, 3, padding=1)  # out: 8
        self.conv_3_1 = nn.Conv3d(128, 128, 3, padding=1)  # out: 8

        # feature_size = (1 +  16 + 32 + 64 + 128 + 128 ) * 7
        feature_size = (input_dim + self.config.IF_IN_DIM + 32 + 64 + 128 + 128 ) * 7
        self.fc_0 = nn.Conv1d(feature_size, hidden_dim, 1)
        self.fc_1 = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.fc_2 = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.fc_out = nn.Conv1d(hidden_dim, 1, 1)
        self.actvn = nn.ReLU()

        self.maxpool = nn.MaxPool3d(2)
        
        self.use_bn = self.config.IF_BN
        if self.use_bn:
            self.conv_in_bn = nn.BatchNorm3d(16)
            self.conv0_1_bn = nn.BatchNorm3d(32)
            self.conv1_1_bn = nn.BatchNorm3d(64)
            self.conv2_1_bn = nn.BatchNorm3d(128)
            self.conv3_1_bn = nn.BatchNorm3d(128)


        displacment = 0.0722
        displacments = []
        displacments.append([0, 0, 0])
        for x in range(3):
            for y in [-1, 1]:
                input = [0, 0, 0]
                input[x] = y * displacment
                displacments.append(input)

        # self.displacments = torch.Tensor(displacments).cuda(GPUnum)
        self.displacments = torch.Tensor(displacments).to(device)


    def forward(self, p, x):
        # x = x.unsqueeze(1)

        p_features = p.transpose(1, -1)
        p = p.unsqueeze(1).unsqueeze(1)
        p = torch.cat([p + d for d in self.displacments], dim=2)  # (B,1,7,num_samples,3)
        feature_0 = F.grid_sample(x, p)  # out : (B,C (of x), 1,1,sample_num)

        net = self.actvn(self.conv_in(x))
        if self.use_bn:
            net = self.conv_in_bn(net)
        feature_1 = F.grid_sample(net, p)  # out : (B,C (of x), 1,1,sample_num)
        net = self.maxpool(net)

        net = self.actvn(self.conv_0(net))
        net = self.actvn(self.conv_0_1(net))
        if self.use_bn:
            net = self.conv0_1_bn(net)
        feature_2 = F.grid_sample(net, p)  # out : (B,C (of x), 1,1,sample_num)
        net = self.maxpool(net)

        net = self.actvn(self.conv_1(net))
        net = self.actvn(self.conv_1_1(net))
        if self.use_bn:
            net = self.conv1_1_bn(net)
        feature_3 = F.grid_sample(net, p)  # out : (B,C (of x), 1,1,sample_num)
        net = self.maxpool(net)

        net = self.actvn(self.conv_2(net))
        net = self.actvn(self.conv_2_1(net))
        if self.use_bn:
            net = self.conv2_1_bn(net)
        feature_4 = F.grid_sample(net, p)
        net = self.maxpool(net)

        net = self.actvn(self.conv_3(net))
        net = self.actvn(self.conv_3_1(net))
        if self.use_bn:
            net = self.conv3_1_bn(net)
        feature_5 = F.grid_sample(net, p)

        # here every channel corresponse to one feature.

        features = torch.cat((feature_0, feature_1, feature_2, feature_3, feature_4, feature_5),
                             dim=1)  # (B, features, 1,7,sample_num)
        shape = features.shape
        features = torch.reshape(features,
                                 (shape[0], shape[1] * shape[3], shape[4]))  # (B, featues_per_sample, samples_num)
        #features = torch.cat((features, p_features), dim=1)  # (B, featue_size, samples_num)

        net = self.actvn(self.fc_0(features))
        net = self.actvn(self.fc_1(net))
        net = self.actvn(self.fc_2(net))
        net = self.fc_out(net)
        out = net.squeeze(1)

        return out