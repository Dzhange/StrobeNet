# Borrowed from https://github.com/meetshah1995/pytorch-semseg
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from torch.autograd import Variable


class conv2DBatchNorm(nn.Module):
    def __init__(
            self,
            in_channels,
            n_filters,
            k_size,
            stride,
            padding,
            bias=True,
            dilation=1,
            is_batchnorm=True,
    ):
        super(conv2DBatchNorm, self).__init__()

        conv_mod = nn.Conv2d(
            int(in_channels),
            int(n_filters),
            kernel_size=k_size,
            padding=padding,
            stride=stride,
            bias=bias,
            dilation=dilation,
        )

        if is_batchnorm:
            self.cb_unit = nn.Sequential(conv_mod, nn.BatchNorm2d(int(n_filters)))
        else:
            self.cb_unit = nn.Sequential(conv_mod)

    def forward(self, inputs):
        outputs = self.cb_unit(inputs)
        return outputs


class conv2DGroupNorm(nn.Module):
    def __init__(
            self, in_channels, n_filters, k_size, stride, padding, bias=True, dilation=1, n_groups=16
    ):
        super(conv2DGroupNorm, self).__init__()

        conv_mod = nn.Conv2d(
            int(in_channels),
            int(n_filters),
            kernel_size=k_size,
            padding=padding,
            stride=stride,
            bias=bias,
            dilation=dilation,
        )

        self.cg_unit = nn.Sequential(conv_mod, nn.GroupNorm(n_groups, int(n_filters)))

    def forward(self, inputs):
        outputs = self.cg_unit(inputs)
        return outputs


class deconv2DBatchNorm(nn.Module):
    def __init__(self, in_channels, n_filters, k_size, stride, padding, bias=True):
        super(deconv2DBatchNorm, self).__init__()

        self.dcb_unit = nn.Sequential(
            nn.ConvTranspose2d(
                int(in_channels),
                int(n_filters),
                kernel_size=k_size,
                padding=padding,
                stride=stride,
                bias=bias,
            ),
            nn.BatchNorm2d(int(n_filters)),
        )

    def forward(self, inputs):
        outputs = self.dcb_unit(inputs)
        return outputs


class conv2DBatchNormRelu(nn.Module):
    def __init__(
            self,
            in_channels,
            n_filters,
            k_size,
            stride,
            padding,
            bias=True,
            dilation=1,
            is_batchnorm=True,
    ):
        super(conv2DBatchNormRelu, self).__init__()

        conv_mod = nn.Conv2d(
            int(in_channels),
            int(n_filters),
            kernel_size=k_size,
            padding=padding,
            stride=stride,
            bias=bias,
            dilation=dilation,
        )

        if is_batchnorm:
            self.cbr_unit = nn.Sequential(
                conv_mod, nn.BatchNorm2d(int(n_filters)), nn.ReLU(inplace=True)
            )
        else:
            self.cbr_unit = nn.Sequential(conv_mod, nn.ReLU(inplace=True))

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs


class conv2DGroupNormRelu(nn.Module):
    def __init__(
            self, in_channels, n_filters, k_size, stride, padding, bias=True, dilation=1, n_groups=16
    ):
        super(conv2DGroupNormRelu, self).__init__()

        conv_mod = nn.Conv2d(
            int(in_channels),
            int(n_filters),
            kernel_size=k_size,
            padding=padding,
            stride=stride,
            bias=bias,
            dilation=dilation,
        )

        self.cgr_unit = nn.Sequential(
            conv_mod, nn.GroupNorm(n_groups, int(n_filters)), nn.ReLU(inplace=True)
        )

    def forward(self, inputs):
        outputs = self.cgr_unit(inputs)
        return outputs


class deconv2DBatchNormRelu(nn.Module):
    def __init__(self, in_channels, n_filters, k_size, stride, padding, bias=True, is_batchnorm=True):
        super(deconv2DBatchNormRelu, self).__init__()

        if is_batchnorm:
            self.dcbr_unit = nn.Sequential(
                nn.ConvTranspose2d(
                    int(in_channels),
                    int(n_filters),
                    kernel_size=k_size,
                    padding=padding,
                    stride=stride,
                    bias=bias,
                ),
                nn.BatchNorm2d(int(n_filters)),
                nn.ReLU(inplace=True),
            )
        else:
            self.dcbr_unit = nn.Sequential(
                nn.ConvTranspose2d(
                    int(in_channels),
                    int(n_filters),
                    kernel_size=k_size,
                    padding=padding,
                    stride=stride,
                    bias=bias,
                ),
                nn.ReLU(inplace=True),
            )

    def forward(self, inputs):
        outputs = self.dcbr_unit(inputs)
        return outputs


class segnetDown2(nn.Module):
    def __init__(self, in_size, out_size, withFeatureMap=False, bn=True):
        super(segnetDown2, self).__init__()
        self.conv1 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1, is_batchnorm=bn)
        self.conv2 = conv2DBatchNormRelu(out_size, out_size, 3, 1, 1, is_batchnorm=bn)
        self.maxpool_with_argmax = nn.MaxPool2d(2, 2, return_indices=True)
        self.withFeatureMap = withFeatureMap

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        FeatureMap = outputs
        unpooled_shape = outputs.size()
        outputs, indices = self.maxpool_with_argmax(outputs)
        if self.withFeatureMap:
            return outputs, indices, unpooled_shape, FeatureMap
        return outputs, indices, unpooled_shape, None


class segnetDown3(nn.Module):
    def __init__(self, in_size, out_size, withFeatureMap=False, bn=True):
        super(segnetDown3, self).__init__()
        self.conv1 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1, is_batchnorm=bn)
        self.conv2 = conv2DBatchNormRelu(out_size, out_size, 3, 1, 1, is_batchnorm=bn)
        self.conv3 = conv2DBatchNormRelu(out_size, out_size, 3, 1, 1, is_batchnorm=bn)
        self.maxpool_with_argmax = nn.MaxPool2d(2, 2, return_indices=True)
        self.withFeatureMap = withFeatureMap

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        FeatureMap = outputs
        unpooled_shape = outputs.size()
        outputs, indices = self.maxpool_with_argmax(outputs)
        if self.withFeatureMap:
            return outputs, indices, unpooled_shape, FeatureMap
        return outputs, indices, unpooled_shape, None


class segnetUp2(nn.Module):
    def __init__(self, in_size, out_size, last_layer=False, withSkipConnections=False, bn=True):
        super().__init__()
        self.withSkipConnections = withSkipConnections
        self.unpool = nn.MaxUnpool2d(2, 2)
        if self.withSkipConnections:
            self.conv1 = deconv2DBatchNormRelu(2 * in_size, 2 * in_size, 3, 1, 1, is_batchnorm=bn)
            if last_layer:
                self.conv2 = nn.ConvTranspose2d(in_channels=2 * in_size, out_channels=out_size, kernel_size=3, padding=1, stride=1)
            else:
                self.conv2 = deconv2DBatchNormRelu(2 * in_size, out_size, 3, 1, 1, is_batchnorm=bn)
        else:
            self.conv1 = deconv2DBatchNormRelu(in_size, in_size, 3, 1, 1, is_batchnorm=bn)
            if last_layer:
                self.conv2 = nn.ConvTranspose2d(in_channels=in_size, out_channels=out_size, kernel_size=3, padding=1, stride=1)
            else:
                self.conv2 = deconv2DBatchNormRelu(in_size, out_size, 3, 1, 1, is_batchnorm=bn)

    def forward(self, inputs, indices, output_shape, SkipFeatureMap=None):
        if self.withSkipConnections and SkipFeatureMap is None:
            raise RuntimeError('Created SegNet with skip connections. But no feature map is passed.')

        outputs = self.unpool(input=inputs, indices=indices, output_size=output_shape)
        if self.withSkipConnections:
            outputs = torch.cat((SkipFeatureMap, outputs), 1)
        outputs = self.conv1(outputs)
        outputs = self.conv2(outputs)

        return outputs


class segnetUp3(nn.Module):
    def __init__(self, in_size, out_size, withSkipConnections=False, bn=True):
        super().__init__()
        self.withSkipConnections = withSkipConnections
        self.unpool = nn.MaxUnpool2d(2, 2)
        if self.withSkipConnections:
            self.conv1 = deconv2DBatchNormRelu(2 * in_size, 2 * in_size, 3, 1, 1, is_batchnorm=bn)
            self.conv2 = deconv2DBatchNormRelu(2 * in_size, 2 * in_size, 3, 1, 1, is_batchnorm=bn)
            self.conv3 = deconv2DBatchNormRelu(2 * in_size, out_size, 3, 1, 1, is_batchnorm=bn)
        else:
            self.conv1 = deconv2DBatchNormRelu(in_size, in_size, 3, 1, 1, is_batchnorm=bn)
            self.conv2 = deconv2DBatchNormRelu(in_size, in_size, 3, 1, 1, is_batchnorm=bn)
            self.conv3 = deconv2DBatchNormRelu(in_size, out_size, 3, 1, 1, is_batchnorm=bn)

    def forward(self, inputs, indices, output_shape, SkipFeatureMap=None):
        if self.withSkipConnections and SkipFeatureMap is None:
            raise RuntimeError('Created SegNet with skip connections. But no feature map is passed.')

        outputs = self.unpool(input=inputs, indices=indices, output_size=output_shape)
        if self.withSkipConnections:
            outputs = torch.cat((SkipFeatureMap, outputs), 1)
        outputs = self.conv1(outputs)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        return outputs
