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
class ShallowSVR(nn.Module):


    def __init__(self, config, device, hidden_dim=256):
        super(ShallowSVR, self).__init__()

        self.config = config
        input_dim = config.FEATURE_CHANNELS

        self.conv_in = nn.Conv3d(input_dim, self.config.IF_IN_DIM, 3, padding=1, padding_mode='replicate')  # out: 256 ->m.p. 128
        self.conv_0 = nn.Conv3d(self.config.IF_IN_DIM, 128, 3, padding=1, padding_mode='replicate')  # out: 128
        self.conv_0_1 = nn.Conv3d(128, 128, 3, padding=1, padding_mode='replicate')  # out: 128 ->m.p. 64
        

        feature_size = (input_dim + self.config.IF_IN_DIM + 128) * 7 + 3
        self.fc_0 = nn.Conv1d(feature_size, hidden_dim * 2, 1)
        self.fc_1 = nn.Conv1d(hidden_dim *2, hidden_dim, 1)
        self.fc_2 = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.fc_out = nn.Conv1d(hidden_dim, 1, 1)
        self.actvn = nn.ReLU()

        self.maxpool = nn.MaxPool3d(2)

        ## try not using bn
        self.use_bn = self.config.IF_BN
        if self.use_bn:
            track_stats = True
            self.conv_in_bn = nn.BatchNorm3d(self.config.IF_IN_DIM, track_running_stats=track_stats)            
            self.conv0_1_bn = nn.BatchNorm3d(32, track_running_stats=track_stats)

        displacment = 0.0722
        displacments = []
        displacments.append([0, 0, 0])
        for x in range(3):
            for y in [-1, 1]:
                input = [0, 0, 0]
                input[x] = y * displacment
                displacments.append(input)

        self.displacments = torch.Tensor(displacments).to(device)

    def forward(self, p, x):
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

        # here every channel corresponse to one feature.

        features = torch.cat((feature_0, feature_1, feature_2), dim=1)  # (B, features, 1, 3 ,sample_num)
        shape = features.shape
        features = torch.reshape(features, (shape[0], shape[1] * shape[3], shape[4]))  # (B, featues_per_sample, samples_num)
        features = torch.cat((features, p_features), dim=1)  # (B, featue_size, samples_num)

        net = self.actvn(self.fc_0(features))
        net = self.actvn(self.fc_1(net))
        net = self.actvn(self.fc_2(net))
        net = self.fc_out(net)
        out = net.squeeze(1)

        return out