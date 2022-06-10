# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import argparse

file_dir = os.path.dirname(__file__)  # the directory that options.py resides in


class MonodepthOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Monodepthv2 options")

        # 路径设置。 训练数据集路径， 日志输出路径
        self.parser.add_argument("--data_path",
                                 type=str,
                                 help="path to the training data",
                                 default=os.path.join(file_dir, "kitti_data"))
        self.parser.add_argument("--log_dir",
                                 type=str,
                                 help="log directory",
                                 default=os.path.join(file_dir, "runs"))

        # 训练选项。
        self.parser.add_argument("--model_name",    # 保存模型日志的文件夹的名称，如：mono_model， stereo_model 以及mono+stereo_model
                                 type=str,
                                 help="the name of the folder to save the model in",
                                 default="mdp")
        self.parser.add_argument("--split",         # 使用哪个训练分组
                                 type=str,
                                 help="which training split to use",
                                 choices=["eigen_zhou", "eigen_full", "odom", "benchmark"],
                                 default="eigen_zhou_test")
        self.parser.add_argument("--num_layers",    # resnet层的数量
                                 type=int,
                                 help="number of resnet layers",
                                 default=18,
                                 choices=[18, 34, 50, 101, 152])
        self.parser.add_argument("--dataset",       # 要训练哪种类型的数据集
                                 type=str,
                                 help="dataset to train on",
                                 default="kitti",
                                 choices=["kitti", "kitti_odom", "kitti_depth", "kitti_test"])
        self.parser.add_argument("--png",           # 如果设置，则从原始 KITTI png 文件（而不是 jpgs）训练
                                 help="if set, trains from raw KITTI png files (instead of jpgs)",
                                 action="store_true")
        self.parser.add_argument("--height",        # 输入图像的高度
                                 type=int,
                                 help="input image height",
                                 default=192)
        self.parser.add_argument("--width",         # 输入图像的宽度
                                 type=int,
                                 help="input image width",
                                 default=640)
        self.parser.add_argument("--disparity_smoothness",  # 视差平滑权重
                                 type=float,
                                 help="disparity smoothness weight",
                                 default=1e-3)
        self.parser.add_argument("--scales",        # 其含义是在encoder和decoder时进行4级缩小和放大的多尺度，其倍数[0,1,2,3]分别对应为1, 2, 4, 8。
                                 nargs="+",
                                 type=int,
                                 help="scales used in the loss",
                                 default=[0, 1, 2, 3])
        self.parser.add_argument("--min_depth",     # 最小深度
                                 type=float,
                                 help="minimum depth",
                                 default=0.1)
        self.parser.add_argument("--max_depth",     # 最大深度
                                 type=float,
                                 help="maximum depth",
                                 default=100.0)
        self.parser.add_argument("--use_stereo",    # 如果设置，则使用stereo pair进行训练
                                 help="if set, uses stereo pair for training",
                                 action="store_true")
        self.parser.add_argument("--frame_ids",     # 要加载的帧 0代表当前输入的样本图片，-1则代表当前帧在这个视频系列中的上一帧，1则代表下一帧。
                                 nargs="+",
                                 type=int,
                                 help="frames to load",
                                 default=[0, -1, 1])

        # 优化选项
        self.parser.add_argument("--batch_size",    # 批量大小
                                 type=int,
                                 help="batch size",
                                 default=2)
        self.parser.add_argument("--learning_rate", # 学习率
                                 type=float,
                                 help="learning rate",
                                 default=1e-4)
        self.parser.add_argument("--num_epochs",    # 迭代次数
                                 type=int,
                                 help="number of epochs",
                                 default=20)
        self.parser.add_argument("--scheduler_step_size",   # 学习率调整的步长
                                 type=int,
                                 help="step size of the scheduler",
                                 default=15)

        # 消融选项
        self.parser.add_argument("--v1_multiscale",         # 如果设置，使用monodepthv1多尺度
                                 help="if set, uses monodepth v1 multiscale",
                                 action="store_true")
        self.parser.add_argument("--avg_reprojection",      # 如果设置，则使用平均重投影损失
                                 help="if set, uses average reprojection loss",
                                 action="store_true")
        self.parser.add_argument("--disable_automasking",   # 如果设置，则不进行automasking
                                 help="if set, doesn't do auto-masking",
                                 action="store_true")
        self.parser.add_argument("--predictive_mask",       # 如果设置，则使用 Zhou 等人的预测掩码方案
                                 help="if set, uses a predictive masking scheme as in Zhou et al",
                                 action="store_true")
        self.parser.add_argument("--no_ssim",               # 如果设置，在损失中禁用 ssim
                                 help="if set, disables ssim in the loss",
                                 action="store_true")
        self.parser.add_argument("--weights_init",          # 选择 预训练 / 从头开始
                                 type=str,
                                 help="pretrained or scratch",
                                 default="pretrained",
                                 choices=["pretrained", "scratch"])
        self.parser.add_argument("--pose_model_input",      # 位姿网络得到多少张图像
                                 type=str,
                                 help="how many images the pose network gets",
                                 default="pairs",
                                 choices=["pairs", "all"])
        self.parser.add_argument("--pose_model_type",       # 位姿网络的类型：正常或共享
                                 type=str,
                                 help="normal or shared",
                                 default="separate_resnet",
                                 choices=["posecnn", "separate_resnet", "shared"])

        # 系统选项
        self.parser.add_argument("--no_cuda",               # CUDA不可用
                                 help="if set disables CUDA",
                                 action="store_true")
        self.parser.add_argument("--num_workers",
                                 type=int,                  # dataloader的线程数量
                                 help="number of dataloader workers",
                                 default=12)

        # 加载选项
        self.parser.add_argument("--load_weights_folder",   # 要加载的权重文件夹路径，可以用 ~ 表示当前用户的home路径
                                 type=str,
                                 help="name of model to load")
        self.parser.add_argument("--models_to_load",        # 加载的模型
                                 nargs="+",
                                 type=str,
                                 help="models to load",
                                 default=["encoder", "depth", "pose_encoder", "pose"])

        # 日志选项
        self.parser.add_argument("--log_frequency",         # 每个tensorboard日志之间的batch
                                 type=int,
                                 help="number of batches between each tensorboard log",
                                 default=250)
        self.parser.add_argument("--save_frequency",        # 每次保存之间的 epoch 数
                                 type=int,
                                 help="number of epochs between each save",
                                 default=1)

        # 评估选项
        self.parser.add_argument("--eval_stereo",           # 如果设置，则在stereo模式下评估
                                 help="if set evaluates in stereo mode",
                                 action="store_true")
        self.parser.add_argument("--eval_mono",             # 如果设置，则在mono模式下评估
                                 help="if set evaluates in mono mode",
                                 action="store_true")
        self.parser.add_argument("--disable_median_scaling",    # 如果设置，则在评估中禁用中值缩放
                                 help="if set disables median scaling in evaluation",
                                 action="store_true")
        self.parser.add_argument("--pred_depth_scale_factor",   # 如果设置，将预测乘以这个数字
                                 help="if set multiplies predictions by this number",
                                 type=float,
                                 default=1)
        self.parser.add_argument("--ext_disp_to_eval",      # 要评估的 .npy 差异文件的可选路径
                                 type=str,
                                 help="optional path to a .npy disparities file to evaluate")
        self.parser.add_argument("--eval_split",            # 在哪个拆分上运行评估
                                 type=str,
                                 default="eigen",
                                 choices=[
                                     "eigen", "eigen_benchmark", "benchmark", "odom_9", "odom_10"],
                                 help="which split to run eval on")
        self.parser.add_argument("--save_pred_disps",       # 如果设置，保存预测的差异
                                 help="if set saves predicted disparities",
                                 action="store_true")
        self.parser.add_argument("--no_eval",               # 如果设置，则禁止评估
                                 help="if set disables evaluation",
                                 action="store_true")
        self.parser.add_argument("--eval_eigen_to_benchmark",   # 假设我们正在从 npy 加载特征结果，但我们想使用新的基准进行评估时设置
                                 help="if set assume we are loading eigen results from npy but "
                                      "we want to evaluate using the new benchmark.",
                                 action="store_true")
        self.parser.add_argument("--eval_out_dir",          # 如果设置，会将差异输出到此文件夹
                                 help="if set will output the disparities to this folder",
                                 type=str)
        self.parser.add_argument("--post_process",          # 如果设置，将从原始monodepth执行翻转后处理
                                 help="if set will perform the flipping post processing "
                                      "from the original monodepth paper",
                                 action="store_true")

    def parse(self):
        self.options = self.parser.parse_args()
        return self.options
