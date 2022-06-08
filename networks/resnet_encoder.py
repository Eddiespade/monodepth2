# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models
import torch.utils.model_zoo as model_zoo


class ResNetMultiImageInput(models.ResNet):
    """
    构建具有不同数量输入图像的 resnet 模型。
    改编自 https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """
    def __init__(self, block, layers, num_classes=1000, num_input_images=1):
        super(ResNetMultiImageInput, self).__init__(block, layers)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            num_input_images * 3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # 网络参数 初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 凯明初始化
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                # 常数初始化
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def resnet_multiimage_input(num_layers, pretrained=False, num_input_images=1):
    """
    构造一个 ResNet 模型。
    num_layers (int):           resnet 层数。 必须是 18 或 50
    pretrained (bool)：         如果为真，则返回在 ImageNet 上预训练的模型
    num_input_images (int):     作为输入堆叠的帧数
    """
    # 只能运行 18 或 50 层 resnet
    assert num_layers in [18, 50], "Can only run with 18 or 50 layer resnet"
    # 如果 num_layers = 18 则 blocks = [2, 2, 2, 2]
    blocks = {18: [2, 2, 2, 2],
              50: [3, 4, 6, 3]}[num_layers]
    # 主干模型
    block_type = {18: models.resnet.BasicBlock, 50: models.resnet.Bottleneck}[num_layers]
    # 源码中 resnet50 的构建： _resnet("resnet50", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)
    model = ResNetMultiImageInput(block_type, blocks, num_input_images=num_input_images)

    # 是否预训练
    if pretrained:
        # model_zoo:有大量的在大数据集上预训练的可供下载的模型;
        loaded = model_zoo.load_url(models.resnet.model_urls['resnet{}'.format(num_layers)])
        # 因为采用了不同数量输入图像，在重写的resnet模型中仅在 conv1 有所不同， conv1 的输入为 3 * num_input_images; 原始的为 3
        loaded['conv1.weight'] = torch.cat(
            [loaded['conv1.weight']] * num_input_images, 1) / num_input_images
        model.load_state_dict(loaded)
    return model


class ResnetEncoder(nn.Module):
    """
    Pytorch module for a resnet encoder
    num_layers: resnet 层数
    pretrained：是否需要预训练
    num_input_images： 输入图片的数量
    """
    def __init__(self, num_layers, pretrained, num_input_images=1):
        super(ResnetEncoder, self).__init__()

        # resnet encoder的每个self.features输出通道，用于构建 depth 网络
        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

        # 判断输入的num_layer 是否 是有效的
        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        # 如果输入图片多于1张，则需要重写resnet
        if num_input_images > 1:
            self.encoder = resnet_multiimage_input(num_layers, pretrained, num_input_images)
        else:
            self.encoder = resnets[num_layers](pretrained)

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

    def forward(self, input_image):
        self.features = []
        x = (input_image - 0.45) / 0.225    # 归一化 输入图像
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        self.features.append(self.encoder.relu(x))
        self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))
        self.features.append(self.encoder.layer2(self.features[-1]))
        self.features.append(self.encoder.layer3(self.features[-1]))
        self.features.append(self.encoder.layer4(self.features[-1]))

        return self.features
