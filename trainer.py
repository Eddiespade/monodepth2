# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import json

from utils import *
from kitti_utils import *
from layers import *

import datasets
import networks
from IPython import embed


class Trainer:
    def __init__(self, options):
        # 训练设置，为namespace对象
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        # 检查高度和宽度是否为 32 的倍数
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}
        self.parameters_to_train = []

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")

        # 设置的损失scale 个数
        self.num_scales = len(self.opt.scales)
        # 输入帧 个数
        self.num_input_frames = len(self.opt.frame_ids)
        # 如果为 位姿模型得到的输入为 pairs 则 位姿帧 个数为 2
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames

        # frame_ids 必须以 0 开头; 即第一个必须为当前输入的样本图像
        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        # 仅使用 stereo训练的时候  self.use_pose_net = True
        self.use_pose_net = not (self.opt.use_stereo and self.opt.frame_ids == [0])

        # 如果使用 含有stereo训练，帧数的id追加 "s"; "s"代表双目当前帧的另一侧的图片；
        if self.opt.use_stereo:
            self.opt.frame_ids.append("s")

        """ 
        ---------------------------------------------- 搭建网络结构 ------------------------------------------------------
        self.models["encoder"]：     编码层，基于resnet搭建
        self.models["depth"]：       depth网络把得到的四种尺度图像输入encoder，得到futures再输入depth_decoder。整个网络类似于U-NET结构。
        self.models["pose_encoder"]：相机位姿网络的编码层
        self.models["pose"]：        相机位姿网络的解码层
        """
        # 搭建encoder模块
        self.models["encoder"] = networks.ResnetEncoder(
            self.opt.num_layers, self.opt.weights_init == "pretrained")
        self.models["encoder"].to(self.device)
        # parameters里存的就是weight，parameters()会返回一个生成器（迭代器）
        self.parameters_to_train += list(self.models["encoder"].parameters())

        # 搭建depth网络
        self.models["depth"] = networks.DepthDecoder(
            self.models["encoder"].num_ch_enc, self.opt.scales)
        self.models["depth"].to(self.device)
        self.parameters_to_train += list(self.models["depth"].parameters())

        # 搭建pose网络
        # 仅使用 stereo 训练时
        if self.use_pose_net:

            # 当pose网络采用独立的resnet时，重新搭建encoder
            if self.opt.pose_model_type == "separate_resnet":
                self.models["pose_encoder"] = networks.ResnetEncoder(
                    self.opt.num_layers,
                    self.opt.weights_init == "pretrained",
                    num_input_images=self.num_pose_frames)
                self.models["pose_encoder"].to(self.device)
                self.parameters_to_train += list(self.models["pose_encoder"].parameters())

                # 搭建pose网络的decoder层
                self.models["pose"] = networks.PoseDecoder(
                    self.models["pose_encoder"].num_ch_enc,
                    num_input_features=1,
                    num_frames_to_predict_for=2)

            # 当采用共享的encoder时，直接使用depth网络搭建的encoder层
            elif self.opt.pose_model_type == "shared":
                self.models["pose"] = networks.PoseDecoder(
                    self.models["encoder"].num_ch_enc, self.num_pose_frames)

            # 当采用posecnn时，重新搭建pose网络
            elif self.opt.pose_model_type == "posecnn":
                self.models["pose"] = networks.PoseCNN(
                    # 输入帧数如果是”pair“ 则为 2; 否则为 len(self.opt.frame_ids) 通常为 3，仅在 stereo训练时为 1
                    self.num_input_frames if self.opt.pose_model_input == "all" else 2)

            self.models["pose"].to(self.device)
            self.parameters_to_train += list(self.models["pose"].parameters())

        # 使用 Zhou 等人的预测掩码方案， 此时需要禁用 automasking
        if self.opt.predictive_mask:
            assert self.opt.disable_automasking, \
                "When using predictive_mask, please disable automasking with --disable_automasking"

            # 作者的预测掩蔽基线的实现与其深度解码器具有相同的架构，并为每个源帧预测一个单独的掩码。
            self.models["predictive_mask"] = networks.DepthDecoder(
                self.models["encoder"].num_ch_enc, self.opt.scales,
                num_output_channels=(len(self.opt.frame_ids) - 1))
            self.models["predictive_mask"].to(self.device)
            self.parameters_to_train += list(self.models["predictive_mask"].parameters())

        # 采用 Adam 优化器对模型参数进行优化训练
        # ----------------------------------------------------------------------------------------------------
        # 语法： torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)[source]
        #     params (iterable)：                 待优化参数的iterable或者是定义了参数组的dict
        #     lr (float, 可选)：                   学习率（默认：1e-3）
        #     betas (Tuple[float, float], 可选)：  用于计算梯度以及梯度平方的运行平均值的系数（默认：0.9，0.999）
        #     eps (float, 可选)：                  为了增加数值计算的稳定性而加到分母里的项（默认：1e-8）
        #     weight_decay (float, 可选)：         权重衰减（L2惩罚）（默认: 0）
        # ----------------------------------------------------------------------------------------------------
        self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate)
        # 模型学习率调整策略
        # ----------------------------------------------------------------------------------------------------
        # 语法： torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma=0.1, last_epoch=-1)
        #     optimizer （Optimizer）：要更改学习率的优化器；
        #     step_size（int）：       每训练step_size个epoch，更新一次参数；
        #     gamma（float）：         更新lr的乘法因子；
        #     last_epoch （int）：     最后一个epoch的index，如果是训练了很多个epoch后中断了，继续训练，这个值就等于
        #                            加载的模型的epoch。默认为-1表示从头开始训练，即从epoch=1开始。
        # ----------------------------------------------------------------------------------------------------
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer, self.opt.scheduler_step_size, 0.1)

        # 加载预训练权重，包括模型和Adam优化器
        if self.opt.load_weights_folder is not None:
            self.load_model()

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)

        """--------------------------------------------- 加载数据集 ---------------------------------------------------"""
        # 主要用到了两种不同的数据集：KITTI raw data(原始数据) 和 KITTI odometry(里程计)
        datasets_dict = {"kitti": datasets.KITTIRAWDataset,
                         "kitti_odom": datasets.KITTIOdomDataset}
        self.dataset = datasets_dict[self.opt.dataset]

        # os.path.dirname(__file__):  返回当前.py文件的上一级目录的绝对路径。
        fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")

        # 读取训练集和验证集文件
        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))
        img_ext = '.png' if self.opt.png else '.jpg'

        # 训练样本的数量
        num_train_samples = len(train_filenames)
        # 训练数据集生成器产生的总步数。epoch × （每个epoch训练的）
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

        train_dataset = self.dataset(
            self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=True, img_ext=img_ext)
        # ----------------------------------------------------------------------------------------------------
        # 语法: DataLoader(dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None, num_workers=0,
        #            collate_fn=None, pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None, *,
        #            prefetch_factor=2, persistent_workers=False)
        # 主要参数：
        #        dataset：       必须首先使用数据集构造 DataLoader 类。
        #        Shuffle ：      是否重新整理数据。
        #        Sampler ：      指的是可选的 torch.utils.data.Sampler 类实例。采样器定义了检索样本的策略，顺序或随机或任何其他方式。
        #                        使用采样器时应将 Shuffle 设置为 false。
        #        Batch_Sampler ：批处理级别。
        #        num_workers ：  加载数据所需的子进程数。
        #        collate_fn ：   将样本整理成批次。Torch 中可以进行自定义整理。
        #        pin_memory：    锁页内存。pin_memory=True，意味着生成的Tensor数据最开始是属于内存中的锁页内存（显存），这样将内存的Tensor转义到GPU的显存就会更快一些
        #                        不锁页内存在主机内存不足时，数据会存放在虚拟内存（硬盘）中。锁页内存存放的内容在任何情况下都不会与主机的虚拟内存进行交换
        #        drop_last ：    告诉如何处理数据集长度除于batch_size余下的数据。True就抛弃，否则保留
        # ----------------------------------------------------------------------------------------------------
        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        # 加载验证集
        val_dataset = self.dataset(
            self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=False, img_ext=img_ext)
        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        self.val_iter = iter(self.val_loader)

        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))   # 定义logs文件位置

        # 使用ssim 损失衡量 重构图像和原始图像之间的差异
        if not self.opt.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)

        # 保存不同尺度下，将深度图像投影为点云的网络层
        self.backproject_depth = {}
        # 保存不同尺度下，将3D点云投影到相机中的网络层
        self.project_3d = {}
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset), len(val_dataset)))

        self.save_opts()

    def set_train(self):
        """
        将所有模型转换为训练模式
        """
        for m in self.models.values():
            m.train()

    def set_eval(self):
        """
        将所有模型转换为测试/验证模式
        """
        for m in self.models.values():
            m.eval()

    def train(self):
        """
        运行整个训练过程
        """
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        for self.epoch in range(self.opt.num_epochs):
            self.run_epoch()
            # 保存模型
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()

    def run_epoch(self):
        """
        运行每个Epoch的训练和验证
        """
        self.model_lr_scheduler.step()          # 更新学习率

        print("Training")
        self.set_train()                        # 将所有模型转换为训练模式

        for batch_idx, inputs in enumerate(self.train_loader):

            before_op_time = time.time()
            # 通过网络传递一个小批量并生成图像和损失
            outputs, losses = self.process_batch(inputs)

            # 将优化器的梯度置0
            self.model_optimizer.zero_grad()
            # 损失反向传播
            losses["loss"].backward()
            # 更新参数
            self.model_optimizer.step()

            duration = time.time() - before_op_time

            # 在 2000 步后减少日志记录以节省时间和磁盘空间
            early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 2000  # 前2000步
            late_phase = self.step % 2000 == 0      # 2000的倍数

            if early_phase or late_phase:
                self.log_time(batch_idx, duration, losses["loss"].cpu().data)

                if "depth_gt" in inputs:
                    self.compute_depth_losses(inputs, outputs, losses)

                self.log("train", inputs, outputs, losses)
                self.val()

            self.step += 1

    def process_batch(self, inputs):
        """
        通过网络传递一个小批量并生成图像和损失
        网络的input最终是大小为42的字典，是通过CPU进行运算的。
        这里的42 = 4(输入的4张图) × 4(4个尺度) × 2(数据增强) + 4(K矩阵4个尺度) × 2(加上逆矩阵) + 2
        """
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

        if self.opt.pose_model_type == "shared":
            # 如果我们对深度和姿态都使用共享encoder（如 monodepthv1 中所提倡的），那么所有图像都通过深度编码器单独前向传播。
            all_color_aug = torch.cat([inputs[("color_aug", i, 0)] for i in self.opt.frame_ids])
            all_features = self.models["encoder"](all_color_aug)
            all_features = [torch.split(f, self.opt.batch_size) for f in all_features]

            features = {}
            for i, k in enumerate(self.opt.frame_ids):
                features[k] = [f[i] for f in all_features]

            outputs = self.models["depth"](features[0])
        else:
            # 否则，只通过深度编码器输入 frame_id 为 0 的图像
            features = self.models["encoder"](inputs["color_aug", 0, 0])
            outputs = self.models["depth"](features)

        if self.opt.predictive_mask:
            outputs["predictive_mask"] = self.models["predictive_mask"](features)

        if self.use_pose_net:
            outputs.update(self.predict_poses(inputs, features))

        self.generate_images_pred(inputs, outputs)
        losses = self.compute_losses(inputs, outputs)

        return outputs, losses

    def predict_poses(self, inputs, features):
        """
        预测单目序列的输入帧之间的位姿。
        """
        outputs = {}
        # 在此设置中，我们通过姿态网络的单独前向传递计算每个源帧的姿态。
        if self.num_pose_frames == 2:
            # 选择姿态网络作为输入的特征
            if self.opt.pose_model_type == "shared":
                pose_feats = {f_i: features[f_i] for f_i in self.opt.frame_ids}
            else:
                pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}

            for f_i in self.opt.frame_ids[1:]:
                if f_i != "s":
                    # 为了保持顺序，总是按时间顺序传递帧
                    if f_i < 0:
                        pose_inputs = [pose_feats[f_i], pose_feats[0]]
                    else:
                        pose_inputs = [pose_feats[0], pose_feats[f_i]]

                    if self.opt.pose_model_type == "separate_resnet":
                        pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]
                    elif self.opt.pose_model_type == "posecnn":
                        pose_inputs = torch.cat(pose_inputs, 1)

                    # 通过pose解码器得到输出的 轴角 和 平移
                    axisangle, translation = self.models["pose"](pose_inputs)
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation

                    # 如果帧 id 为负，则反转矩阵
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

        else:   # 将所有帧一起输入到姿态网络（并预测所有姿态）
            if self.opt.pose_model_type in ["separate_resnet", "posecnn"]:
                pose_inputs = torch.cat(
                    [inputs[("color_aug", i, 0)] for i in self.opt.frame_ids if i != "s"], 1)

                if self.opt.pose_model_type == "separate_resnet":
                    pose_inputs = [self.models["pose_encoder"](pose_inputs)]

            elif self.opt.pose_model_type == "shared":
                pose_inputs = [features[i] for i in self.opt.frame_ids if i != "s"]

            axisangle, translation = self.models["pose"](pose_inputs)

            for i, f_i in enumerate(self.opt.frame_ids[1:]):
                if f_i != "s":
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, i], translation[:, i])

        return outputs

    def val(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()
        try:
            inputs = self.val_iter.next()
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = self.val_iter.next()

        with torch.no_grad():
            outputs, losses = self.process_batch(inputs)

            if "depth_gt" in inputs:
                self.compute_depth_losses(inputs, outputs, losses)

            self.log("val", inputs, outputs, losses)
            del inputs, outputs, losses

        self.set_train()

    def generate_images_pred(self, inputs, outputs):
        """
        为小批量生成扭曲（重新投影）的彩色图像。
        生成的图像被保存到 `outputs` 字典中。
        """
        # outputs["disp"]直接输出的就是视差图，并且仍然多尺度[0,1,2,3]分布。
        for scale in self.opt.scales:
            disp = outputs[("disp", scale)]
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                # ------------------------------------------------------------------------------------------------------
                # 语法：torch.nn.functional.interpolate(input, size=None, scale_factor=None, mode='nearest',
                #                                       align_corners=None, recompute_scale_factor=None)
                # 详解参考： https://blog.csdn.net/qq_50001789/article/details/120297401
                # ------------------------------------------------------------------------------------------------------
                disp = F.interpolate(
                    disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                source_scale = 0

            # 将disp值映射到[0.01,10]，并求倒数就能得到深度值
            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
            # 将深度值存放到outputs["depth"...]中
            outputs[("depth", 0, scale)] = depth

            # 在stereo 训练时， frame_id恒为"s"。
            for i, frame_id in enumerate(self.opt.frame_ids[1:]):

                if frame_id == "s":
                    T = inputs["stereo_T"]
                else:
                    T = outputs[("cam_T_cam", 0, frame_id)]

                # from the authors of https://arxiv.org/abs/1712.00175
                if self.opt.pose_model_type == "posecnn":

                    axisangle = outputs[("axisangle", 0, frame_id)]
                    translation = outputs[("translation", 0, frame_id)]

                    inv_depth = 1 / depth
                    mean_inv_depth = inv_depth.mean(3, True).mean(2, True)

                    T = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0] * mean_inv_depth[:, 0], frame_id < 0)
                # 将深度图投影成3维点云
                cam_points = self.backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)])
                # 将3维点云投影成二维图像
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T)
                # 将二维图像赋值给outputs[("sample"..)]
                outputs[("sample", frame_id, scale)] = pix_coords

                # outputs上某点(x,y)的三个通道像素值来自于inputs上的(x',y'); 而x'和y'则由outputs(x,y)的最低维[0]和[1].
                # grid_sample(input, grid, mode = "bilinear", padding_mode = "zeros", align_corners = None)：
                #           提供一个input以及一个网格，然后根据grid中每个位置提供的坐标信息(input中pixel的坐标)，
                #           将input中对应位置的像素值填充到grid指定的位置，得到最终的输出。
                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border")  # padding_mode="border": 对于越界的位置在⽹格中采⽤边界的pixel value进⾏填充。

                if not self.opt.disable_automasking:
                    outputs[("color_identity", frame_id, scale)] = \
                        inputs[("color", frame_id, source_scale)]

    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    def compute_losses(self, inputs, outputs):
        """
        计算小批量的重投影和平滑损失
        """
        losses = {}
        total_loss = 0

        # 按尺度来计算loss
        for scale in self.opt.scales:
            loss = 0
            reprojection_losses = []

            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                source_scale = 0

            disp = outputs[("disp", scale)]                 # 按尺度获得视差图
            color = inputs[("color", 0, scale)]             # 按尺度获得原始输入图
            target = inputs[("color", 0, source_scale)]     # 0 尺度的原始输入图

            # 在stereo训练时，frame_id恒为“s”
            for frame_id in self.opt.frame_ids[1:]:
                # 按尺度获得对应图像的预测图（即深度图转换到点云再转到二维图像最后采样得到的彩图
                pred = outputs[("color", frame_id, scale)]
                # 根据pred多尺度图和0尺度
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))

            reprojection_losses = torch.cat(reprojection_losses, 1)

            # 直接对inputs["color",0,0]和["color",s,0]计算identity loss
            if not self.opt.disable_automasking:
                identity_reprojection_losses = []
                for frame_id in self.opt.frame_ids[1:]:
                    pred = inputs[("color", frame_id, source_scale)]
                    identity_reprojection_losses.append(
                        self.compute_reprojection_loss(pred, target))

                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

                if self.opt.avg_reprojection:
                    identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
                else:
                    # save both images, and do min all at once below
                    identity_reprojection_loss = identity_reprojection_losses

            elif self.opt.predictive_mask:
                # use the predicted mask
                mask = outputs["predictive_mask"]["disp", scale]
                if not self.opt.v1_multiscale:
                    mask = F.interpolate(
                        mask, [self.opt.height, self.opt.width],
                        mode="bilinear", align_corners=False)

                reprojection_losses *= mask

                # add a loss pushing mask to 1 (using nn.BCELoss for stability)
                weighting_loss = 0.2 * nn.BCELoss()(mask, torch.ones(mask.shape).cuda())
                loss += weighting_loss.mean()

            if self.opt.avg_reprojection:
                reprojection_loss = reprojection_losses.mean(1, keepdim=True)
            else:
                reprojection_loss = reprojection_losses

            if not self.opt.disable_automasking:
                # add random numbers to break ties
                identity_reprojection_loss += torch.randn(
                    identity_reprojection_loss.shape, device=self.device) * 0.00001

                combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
            else:
                combined = reprojection_loss

            if combined.shape[1] == 1:
                to_optimise = combined
            else:
                to_optimise, idxs = torch.min(combined, dim=1)

            if not self.opt.disable_automasking:
                outputs["identity_selection/{}".format(scale)] = (
                    idxs > identity_reprojection_loss.shape[1] - 1).float()

            loss += to_optimise.mean()

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)

            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
            total_loss += loss
            losses["loss/{}".format(scale)] = loss

        total_loss /= self.num_scales
        losses["loss"] = total_loss
        return losses

    def compute_depth_losses(self, inputs, outputs, losses):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        depth_pred = outputs[("depth", 0, 0)]
        depth_pred = torch.clamp(F.interpolate(
            depth_pred, [375, 1242], mode="bilinear", align_corners=False), 1e-3, 80)
        depth_pred = depth_pred.detach()

        depth_gt = inputs["depth_gt"]
        mask = depth_gt > 0

        # garg/eigen crop
        crop_mask = torch.zeros_like(mask)
        crop_mask[:, :, 153:371, 44:1197] = 1
        mask = mask * crop_mask

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

        depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)

        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = np.array(depth_errors[i].cpu())

    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

        for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
            for s in self.opt.scales:
                for frame_id in self.opt.frame_ids:
                    writer.add_image(
                        "color_{}_{}/{}".format(frame_id, s, j),
                        inputs[("color", frame_id, s)][j].data, self.step)
                    if s == 0 and frame_id != 0:
                        writer.add_image(
                            "color_pred_{}_{}/{}".format(frame_id, s, j),
                            outputs[("color", frame_id, s)][j].data, self.step)

                writer.add_image(
                    "disp_{}/{}".format(s, j),
                    normalize_image(outputs[("disp", s)][j]), self.step)

                if self.opt.predictive_mask:
                    for f_idx, frame_id in enumerate(self.opt.frame_ids[1:]):
                        writer.add_image(
                            "predictive_mask_{}_{}/{}".format(frame_id, s, j),
                            outputs["predictive_mask"][("disp", s)][j, f_idx][None, ...],
                            self.step)

                elif not self.opt.disable_automasking:
                    writer.add_image(
                        "automask_{}/{}".format(s, j),
                        outputs["identity_selection/{}".format(s)][j][None, ...], self.step)

    def save_opts(self):
        """
        将option的所有参数保存，便于后续复现
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        # 拷贝 opt字典
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            # json.dumps()将一个Python数据结构转换为JSON; indent: 参数根据数据格式缩进显示，读起来更加清晰。
            json.dump(to_save, f, indent=2)

    def save_model(self):
        """
        保存模型权重参数，不是在原有的基础上更新，而是重新保存一个新的权重参数文件
        """
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        # 保存模型参数
        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'encoder':
                # 保存尺寸 - 在预测时需要这些尺寸
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
                to_save['use_stereo'] = self.opt.use_stereo
            torch.save(to_save, save_path)

        # 保存优化器参数
        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)

    def load_model(self):
        """
        从磁盘加载模型
        """
        # os.path.expanduser(): 它可以将参数中开头部分的 ~ 或 ~user 替换为当前用户的home目录并返回, 即展开~
        # windows： C:\\Users\\user_name   ------   linux: /home/user_name
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        # 权重路径 必须是 文件夹
        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        # 加载每个网络的预训练权重
        for n in self.opt.models_to_load:               # ["encoder", "depth", "pose_encoder", "pose"]
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)          # 从文件中加载一个用torch.save()保存的对象。
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)          # 把预训练有的网络层的参数更新进来
            self.models[n].load_state_dict(model_dict)  # 加载到网络

        # 加载Adam权重
        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            self.model_optimizer.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")
