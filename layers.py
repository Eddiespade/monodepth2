# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def disp_to_depth(disp, min_depth, max_depth):
    """
    将网络的 sigmoid 输出转换为深度预测，此转换的公式在论文的“其他注意事项”部分中给出。
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth


def transformation_from_parameters(axisangle, translation, invert=False):
    """
    将网络的 (axisangle, translation) 输出转换为 4x4 矩阵
    """
    # R 为将轴角旋转转换得到的 4x4 变换矩阵
    R = rot_from_axisangle(axisangle)
    t = translation.clone()

    # 是否需要反转矩阵
    if invert:
        R = R.transpose(1, 2)
        t *= -1

    T = get_translation_matrix(t)

    if invert:
        M = torch.matmul(R, T)
    else:
        M = torch.matmul(T, R)

    return M


def get_translation_matrix(translation_vector):
    """
    将平移向量转换为 4x4 变换矩阵
    """
    T = torch.zeros(translation_vector.shape[0], 4, 4).to(device=translation_vector.device)
    # .contiguous()： 把tensor变成在内存中连续分布的形式。
    t = translation_vector.contiguous().view(-1, 3, 1)

    T[:, 0, 0] = 1
    T[:, 1, 1] = 1
    T[:, 2, 2] = 1
    T[:, 3, 3] = 1
    # None： 在保证数据不改变的情况下,追加一个新纬度.
    T[:, :3, 3, None] = t

    return T


def rot_from_axisangle(vec):
    """
    将轴角旋转转换为 4x4 变换矩阵（改编自 https://github.com/Wallacoloo/printipi）
    输入“vec”必须是 Bx1x3
    """
    # torch.norm(input, p, dim, keepdim=False, out=None) → Tensor： 求指定维度上的范数
    angle = torch.norm(vec, 2, 2, True)
    axis = vec / (angle + 1e-7)

    ca = torch.cos(angle)
    sa = torch.sin(angle)
    C = 1 - ca

    x = axis[..., 0].unsqueeze(1)
    y = axis[..., 1].unsqueeze(1)
    z = axis[..., 2].unsqueeze(1)

    xs = x * sa
    ys = y * sa
    zs = z * sa
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    rot = torch.zeros((vec.shape[0], 4, 4)).to(device=vec.device)

    rot[:, 0, 0] = torch.squeeze(x * xC + ca)
    rot[:, 0, 1] = torch.squeeze(xyC - zs)
    rot[:, 0, 2] = torch.squeeze(zxC + ys)
    rot[:, 1, 0] = torch.squeeze(xyC + zs)
    rot[:, 1, 1] = torch.squeeze(y * yC + ca)
    rot[:, 1, 2] = torch.squeeze(yzC - xs)
    rot[:, 2, 0] = torch.squeeze(zxC - ys)
    rot[:, 2, 1] = torch.squeeze(yzC + xs)
    rot[:, 2, 2] = torch.squeeze(z * zC + ca)
    rot[:, 3, 3] = 1

    return rot


class ConvBlock(nn.Module):
    """
    卷积层，采用 ELU 激活函数
    in_channels:    输入的通道数
    out_channels:   输出的通道数
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out


class Conv3x3(nn.Module):
    """
    在解码器中使用反射填充代替零填充，当样本到达图像边界之外时，返回源图像中最近边界像素的值。显著减少了现有方法中发现的边界伪影
    用来对输入进行填充和卷积操作
    in_channels:    输入的通道数
    out_channels:   输出的通道数
    use_refl：      是否使用反射
    """
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        # 使用反射padding

        # ----------------------------------------------------------------------------------------
        #                        pytorch 的四种填充方法
        # 详解参考：https://blog.csdn.net/weixin_43135178/article/details/117618486
        # 1、零填充ZeroPad2d
        # 2、常数填充ConstantPad2d
        # 3、镜像填充ReflectionPad2d
        # 4、重复填充ReplicationPad2d
        # ----------------------------------------------------------------------------------------

        if use_refl:
            # ------------------------------------------------------------------------------------
            # 详解参考：https://blog.csdn.net/LionZYT/article/details/120181586
            # 函数用途：对输入图像以最外围像素为对称轴，做四周的轴对称镜像填充。
            # 填充顺序：左->右->上->下
            # 对于一个4维的Tensor，当只指定一个padding参数时，则表示对四周采用相同的填充行数。
            # 例子，对四周都填充3行：
            #     nn.ReflectionPad2d(3)
            # ------------------------------------------------------------------------------------
            self.pad = nn.ReflectionPad2d(1)
        else:
            # ------------------------------------------------------------------------------------
            # 对Tensor使用0进行边界填充，我们可以指定tensor的四个方向上的填充数，
            # 比如左边添加1dim、右边添加2dim、上边添加3dim、下边添加4dim，即指定paddin参数为（1，2，3，4），如下：
            #     pad = nn.ZeroPad2d(padding=(1, 2, 3, 4))
            # -------------------------------------------------------------------------------------
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out


class BackprojectDepth(nn.Module):
    """
    将深度图像转换为点云的层
    batch_size：     批量大小
    height：         当前尺度的高度
    width：          当前尺度的宽度
    """
    def __init__(self, batch_size, height, width):
        super(BackprojectDepth, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

        # np.meshgrid()：从坐标向量中返回坐标矩阵; indexing：输出的笛卡尔（默认为“ xy”）或矩阵（“ ij”）索引。
        # 返回list,有两个元素,第一个元素是X轴的取值,第二个元素是Y轴的取值
        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        # 作为nn.Module中的可训练参数使用，但参数不更新
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords),
                                      requires_grad=False)
        # 对应尺寸的全 1 参数
        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
                                 requires_grad=False)

        # 将x, y 所有取值在 0 维拼接，然后再在拼接后的 0维增加一个长度为1的维度
        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        # .repeat(): 对张量进行重复扩充。 详见：https://blog.csdn.net/qq_34806812/article/details/89388210
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1),
                                       requires_grad=False)

    def forward(self, depth, inv_K):
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
        cam_points = depth.view(self.batch_size, 1, -1) * cam_points
        cam_points = torch.cat([cam_points, self.ones], 1)

        return cam_points


class Project3D(nn.Module):
    """
    将 3D 点投影到具有内联函数 K 和位置 T 的相机中的层
    """
    def __init__(self, batch_size, height, width, eps=1e-7):
        super(Project3D, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps

    def forward(self, points, K, T):
        P = torch.matmul(K, T)[:, :3, :]

        cam_points = torch.matmul(P, points)

        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + self.eps)
        pix_coords = pix_coords.view(self.batch_size, 2, self.height, self.width)
        pix_coords = pix_coords.permute(0, 2, 3, 1)
        pix_coords[..., 0] /= self.width - 1
        pix_coords[..., 1] /= self.height - 1
        pix_coords = (pix_coords - 0.5) * 2
        return pix_coords


def upsample(x):
    """
    将输入tensor上采样 2 倍
    """
    # --------------------------------------------------------------------------------------------------
    # 语法：torch.nn.functional.interpolate(input, size=None, scale_factor=None, mode='nearest',
    #                                      align_corners=None, recompute_scale_factor=None)
    # input(Tensor) ：             需要进行采样处理的数组。
    # size(int或序列)：             输出空间的大小
    # scale_factor(float或序列)：   空间大小的乘数
    # mode(str)：                  用于采样的算法。'nearest'| 'linear'| 'bilinear'| 'bicubic'| 'trilinear'
    #       | 'area'。默认：'nearest'
    # align_corners(bool)：        在几何上，我们将输入和输出的像素视为正方形而不是点。如果设置为True，则输入和输出张量
    #       按其角像素的中心点对齐，保留角像素处的值。 如果设置为False，则输入和输出张量通过其角像素的角点对齐，并且插值使
    #       用边缘值填充用于边界外值，使此操作在保持不变时独立于输入大小scale_factor。
    # recompute_scale_facto(bool)：重新计算用于插值计算的 scale_factor。当scale_factor作为参数传递时，
    #       它用于计算output_size。如果recompute_scale_factor的False或没有指定，传入的scale_factor将在插值计算中
    #       使用。否则，将根据用于插值计算的输出和输入大小计算新的scale_factor（即，如果计算的output_size显式传入， 则
    #       计算将相同 ）。注意当scale_factor 是浮点数，由于舍入和精度问题，重新计算的 scale_factor 可能与传入的不同。
    # --------------------------------------------------------------------------------------------------
    return F.interpolate(x, scale_factor=2, mode="nearest")


def get_smooth_loss(disp, img):
    """
    计算视差图像的平滑度损失；彩色图像用于边缘感知平滑度
    """
    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    return grad_disp_x.mean() + grad_disp_y.mean()


class SSIM(nn.Module):
    """
    计算一对图像之间的 Structural Similarity (SSIM) 损失， 用于度量重建后的图片和原图的结构相似性
    详解参考： https://blog.csdn.net/Kevin_cc98/article/details/79028507
    """
    def __init__(self):
        super(SSIM, self).__init__()
        # 语法：  torch.nn.AvgPool2d(kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True)
        # 采用 kernel_size = 3， stride = 1 的窗口做平均池化
        self.mu_x_pool   = nn.AvgPool2d(3, 1)
        self.mu_y_pool   = nn.AvgPool2d(3, 1)
        self.sig_x_pool  = nn.AvgPool2d(3, 1)
        self.sig_y_pool  = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)                            # x的局部均值
        mu_y = self.mu_y_pool(y)                            # y的局部均值

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2      # x的局部方差 D(X)=E(X^2)-E(X)^2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2      # y的局部方差
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y    # 计算协方差  D(XY)=E(X*Y)-E(X)*E(Y)

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        # clamp(input, min, max, out=None) ： 功能将输入input张量每个元素的值压缩到区间 [min,max]，并返回结果到一个新张量。
        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


def compute_depth_errors(gt, pred):
    """
    计算预测和真实标签深度之间的误差度量
    """
    thresh = torch.max((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()

    rmse = (gt - pred) ** 2
    rmse = torch.sqrt(rmse.mean())

    rmse_log = (torch.log(gt) - torch.log(pred)) ** 2
    rmse_log = torch.sqrt(rmse_log.mean())

    abs_rel = torch.mean(torch.abs(gt - pred) / gt)

    sq_rel = torch.mean((gt - pred) ** 2 / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3
