# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import random
import numpy as np
import copy
from PIL import Image  # using pillow-simd for increased speed

import torch
import torch.utils.data as data
from torchvision import transforms


def pil_loader(path):
    # 将路径作为文件打开以避免 ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class MonoDataset(data.Dataset):
    """
    单目数据加载器的父类
        data_path   ：   数据集的父路径
        filenames   ：   txt 文件中的每一行
        height      ：   高度
        width       ：   宽度
        frame_idxs  ：   帧数
        num_scales  ：   多尺度的数量
        is_train    ：   是否是训练集
        img_ext     ：   图片后缀
    """
    def __init__(self,
                 data_path,
                 filenames,
                 height,
                 width,
                 frame_idxs,
                 num_scales,
                 is_train=False,
                 img_ext='.jpg'):
        super(MonoDataset, self).__init__()

        self.data_path = data_path
        self.filenames = filenames
        self.height = height
        self.width = width
        self.num_scales = num_scales
        self.interp = Image.ANTIALIAS

        self.frame_idxs = frame_idxs

        self.is_train = is_train
        self.img_ext = img_ext

        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()

        # We need to specify augmentations differently in newer versions of torchvision.
        # We first try the newer tuple version; if this fails we fall back to scalars
        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1

        self.resize = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize((self.height // s, self.width // s),
                                               interpolation=self.interp)

        self.load_depth = self.check_depth()

    def preprocess(self, inputs, color_aug):
        """
        将彩色图像调整到所需的比例，并且如果需要的话可以对图像进行增强

        我们预先创建了 color_aug 对象，并将相同的增强应用于该项目中的所有图像。
        这确保了输入到姿态网络的所有图像都接收到相同的增强。
        """
        for k in list(inputs):
            frame = inputs[k]
            if "color" in k:
                n, im, i = k
                for i in range(self.num_scales):
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])

        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                n, im, i = k
                inputs[(n, im, i)] = self.to_tensor(f)
                inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        """
        将数据集中的单个训练项作为字典返回。
        值对应于tensor张量。
        字典中的键是字符串或元组：

            ("color", <frame_id>, <scale>)      用于原始彩色图像，
            ("color_aug", <frame_id>, <scale>)  用于增强的彩色图像，
            ("K", scale) 或 ("inv_K", scale)     用于相机内置参数，
            “stereo_T”                          用于相机外部参数
            “depth_gt”                          真实标签的深度图

        <frame_id> 是：
            表示相对于“索引”的时间步长的整数（例如 0、-1 或 1），
        或者
            “s”表示立体对中相对的一侧图像。

        <scale> 是一个整数，表示图像相对于全尺寸图像的比例：
            -1 从磁盘加载的原始分辨率图像
            0 张图片大小调整为 (self.width, self.height)
            1 张图片大小调整为 (self.width // 2, self.height // 2)
            2 张图片大小调整为 (self.width // 4, self.height // 4)
            3 张图片大小调整为 (self.width // 8, self.height // 8)
        """
        inputs = {}

        # 随机做训练数据颜色增强预处理
        do_color_aug = self.is_train and random.random() > 0.5
        # 随机做训练数据水平左右flip预处理
        do_flip = self.is_train and random.random() > 0.5

        # index是train_txt中的第index行。
        line = self.filenames[index].split()
        # train_files.txt中一行数据的第一部分，即图片所在目录。
        folder = line[0]

        # 每一行一般都为3个部分，第二个部分是图片的frame_index
        if len(line) == 3:
            frame_index = int(line[1])
        else:
            frame_index = 0

        # side为l或r，表明该图片是左或右摄像头所拍。
        if len(line) == 3:
            side = line[2]
        else:
            side = None

        # 在stereo训练时， frame_idxs为["0","s"]
        # 通过这个for循环，inputs[("color", "0", -1)]和inputs[("color", "s", -1)]
        # 分别获得了frame_index和它对应的另外一个摄像头拍的图片数据。
        for i in self.frame_idxs:
            if i == "s":
                other_side = {"r": "l", "l": "r"}[side]
                inputs[("color", i, -1)] = self.get_color(folder, frame_index, other_side, do_flip)
            else:
                inputs[("color", i, -1)] = self.get_color(folder, frame_index + i, side, do_flip)

        # 调整内置参数以匹配金字塔（多尺度）中的每个比例
        # 因为模型有4个尺度，所以对应4个相机内参
        for scale in range(self.num_scales):
            K = self.K.copy()

            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)

            inv_K = np.linalg.pinv(K)

            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        # 颜色增强参数设定
        if do_color_aug:
            # color_aug = transforms.ColorJitter.get_params(self.brightness, self.contrast, self.saturation, self.hue)
            color_aug = transforms.ColorJitter(self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)

        # 训练前数据预处理以及对输入数据做多尺度resize。
        self.preprocess(inputs, color_aug)

        # 经过preprocess，产生了inputs[("color"，"0", 0/1/2/3)]和inputs[("color_aug"，"0", 0/1/2/3)]。
        # 所以可以将原始的inputs[("color", i, -1)]和[("color_aug", i, -1)]释放
        for i in self.frame_idxs:
            del inputs[("color", i, -1)]
            del inputs[("color_aug", i, -1)]

        # load_depth为False，因为不需要GT label数据
        if self.load_depth:
            depth_gt = self.get_depth(folder, frame_index, side, do_flip)
            inputs["depth_gt"] = np.expand_dims(depth_gt, 0)
            inputs["depth_gt"] = torch.from_numpy(inputs["depth_gt"].astype(np.float32))

        # 在stereo训练时，还需要构造双目姿态的平移矩阵参数inputs["stereo_T"]
        if "s" in self.frame_idxs:
            stereo_T = np.eye(4, dtype=np.float32)
            baseline_sign = -1 if do_flip else 1
            side_sign = -1 if side == "l" else 1
            stereo_T[0, 3] = side_sign * baseline_sign * 0.1

            inputs["stereo_T"] = torch.from_numpy(stereo_T)

        return inputs

    def get_color(self, folder, frame_index, side, do_flip):
        raise NotImplementedError

    def check_depth(self):
        raise NotImplementedError

    def get_depth(self, folder, frame_index, side, do_flip):
        raise NotImplementedError
