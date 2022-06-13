# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import skimage.transform
import numpy as np
import PIL.Image as pil

from kitti_utils import generate_depth_map
from .mono_dataset import MonoDataset


class KITTIDataset(MonoDataset):
    """
    不同类型的 KITTI 数据集加载器的 父类
    """
    def __init__(self, *args, **kwargs):
        super(KITTIDataset, self).__init__(*args, **kwargs)

        # 注意：确保你的内置参数矩阵可以被原始图像大小归一化。
        # 归一化，您需要将第一行缩放 1 / image_width，第二行缩放 1 / image_height。
        # Monodepth2 假设主点精确居中。 如果您的主要点远离中心，您可能需要禁用水平翻转增强。
        self.K = np.array([[0.58, 0, 0.5, 0],
                           [0, 1.92, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        # 全分辨率尺寸
        self.full_res_shape = (1242, 375)
        # 存放彩色图的文件名映射
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}

    """检查对应的点云信息是否存在"""
    def check_depth(self):
        line = self.filenames[0].split()
        # 当前数据集的路径；如：2011_09_30/2011_09_30_drive_0020_sync
        scene_name = line[0]
        # 帧数；如：185
        frame_index = int(line[1])

        # 找到当前图像对应的三维点云数据；如：kitti_data/2011_09_30/2011_09_30_drive_0020_sync/velodyne_points/data/0000000000.bin
        velo_filename = os.path.join(
            self.data_path,
            scene_name,
            "velodyne_points/data/{:010d}.bin".format(int(frame_index)))
        # 返回是否存在这个文件
        return os.path.isfile(velo_filename)

    """读取彩色图，可以翻转"""
    def get_color(self, folder, frame_index, side, do_flip):
        # folder：文件路径
        # frame_index: 哪一帧
        # side：哪边图像
        # self.loader --> pil_loader():
        color = self.loader(self.get_image_path(folder, frame_index, side))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)    # 对图像进行水平翻转

        return color


class KITTIRAWDataset(KITTIDataset):
    """
    KITTI 数据集，它加载原始 velodyne 深度图以获取地面实况
    """
    def __init__(self, *args, **kwargs):
        super(KITTIRAWDataset, self).__init__(*args, **kwargs)

    """ 获取彩色图片的路径 """
    def get_image_path(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path, folder, "image_0{}/data".format(self.side_map[side]), f_str)
        return image_path

    """ 获取真实深度标签 """
    def get_depth(self, folder, frame_index, side, do_flip):
        calib_path = os.path.join(self.data_path, folder.split("/")[0])

        velo_filename = os.path.join(
            self.data_path,
            folder,
            "velodyne_points/data/{:010d}.bin".format(int(frame_index)))

        # 从 velodyne 数据生成深度图
        depth_gt = generate_depth_map(calib_path, velo_filename, self.side_map[side])
        # 对真实深度进行尺度缩放，会顺便把图片的像素归一化缩放到(0,1)区间内；order=0：最近邻插值，preserve_range=True：保持原来的取值范围。
        depth_gt = skimage.transform.resize(
            depth_gt, self.full_res_shape[::-1], order=0, preserve_range=True, mode='constant')

        if do_flip:
            # 矩阵左右翻转
            depth_gt = np.fliplr(depth_gt)

        return depth_gt


class KITTIOdomDataset(KITTIDataset):
    """
    用于里程计训练和测试的 KITTI 数据集
    """
    def __init__(self, *args, **kwargs):
        super(KITTIOdomDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:06d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path,
            "sequences/{:02d}".format(int(folder)),
            "image_{}".format(self.side_map[side]),
            f_str)
        return image_path


class KITTIDepthDataset(KITTIDataset):
    """
    使用更新的真实标签深度图的 KITTI 数据集
    """
    def __init__(self, *args, **kwargs):
        super(KITTIDepthDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path,
            folder,
            "image_0{}/data".format(self.side_map[side]),
            f_str)
        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        f_str = "{:010d}.png".format(frame_index)
        depth_path = os.path.join(
            self.data_path,
            folder,
            "proj_depth/groundtruth/image_0{}".format(self.side_map[side]),
            f_str)

        depth_gt = pil.open(depth_path)
        depth_gt = depth_gt.resize(self.full_res_shape, pil.NEAREST)
        depth_gt = np.array(depth_gt).astype(np.float32) / 256

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt
