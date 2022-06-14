# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict
from layers import *


class DepthDecoder(nn.Module):
    """
    num_ch_enc:              encoder特征的输出通道数 如resnet18：[64, 64, 128, 256, 512]
    scales：                 损失的尺度
    num_output_channels：    输出通道数
    use_skips：              是否使用跨层连接
    """
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        super(DepthDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips          # 是否采用跨层连接
        self.upsample_mode = 'nearest'      # 上采样的方式
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        # 解码层的通道数
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        # decoder
        # OrderedDict()：按照有序插入顺序存储 的有序字典
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]   # 输入的通道数
            num_ch_out = self.num_ch_dec[i]                                         # 输出的通道数
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)         # 搭建卷积层

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            # 当 i > 0 且 use_skips = True时， 当前decoder卷积层的输入通道需要和 i - 1 的encoder输出通道拼接（跨层拼接）
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        self.outputs = {}

        # decoder
        x = input_features[-1]
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]
            # 如果采用跨层连接  就和 encoder 的对应输出进行拼接
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)

            if i in self.scales:
                self.outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))

        return self.outputs
