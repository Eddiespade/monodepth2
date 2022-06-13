from __future__ import absolute_import, division, print_function

import os
import numpy as np
from collections import Counter


def load_velodyne_points(filename):
    """
    从 KITTI 文件格式加载 3D 点云
    (改编自 https://github.com/hunse/kitti)
    """
    # np.fromfile(): 从文本或二进制文件中的数据构造一个数组。
    points = np.fromfile(filename, dtype=np.float32).reshape(-1, 4)
    points[:, 3] = 1.0  # homogeneous
    return points


def read_calib_file(path):
    """
    读取 KITTI 校准文件
    (来自 https://github.com/hunse/kitti)
    """
    float_chars = set("0123456789.e+- ")
    data = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            # 分割成两个子字符串
            key, value = line.split(':', 1)
            # 移除字符串头尾指定的字符（默认为空格或换行符）或字符序列
            value = value.strip()
            data[key] = value
            # 集合 value 中的所有项目是否都存在于集合 float_chars 中
            if float_chars.issuperset(value):
                # 尝试转换为浮点数组
                try:
                    data[key] = np.array(list(map(float, value.split(' '))))
                except ValueError:
                    # 捕获错误: data[key] already eq. value, so pass
                    pass

    return data


def sub2ind(matrixSize, rowSub, colSub):
    """
    将行、列矩阵下标转换为线性索引
    """
    m, n = matrixSize
    return rowSub * (n-1) + colSub - 1


def generate_depth_map(calib_dir, velo_filename, cam=2, vel_depth=False):
    """
    从 velodyne 数据生成深度图
    """
    # 加载校准文件; 返回字典
    cam2cam = read_calib_file(os.path.join(calib_dir, 'calib_cam_to_cam.txt'))
    velo2cam = read_calib_file(os.path.join(calib_dir, 'calib_velo_to_cam.txt'))
    # np.hstack(): 将参数元组的元素数组按水平方向（列顺序）堆叠数组构成一个新的数组
    velo2cam = np.hstack((velo2cam['R'].reshape(3, 3), velo2cam['T'][..., np.newaxis]))
    # np.vstack(): 将参数元组的元素数组按垂直方向（行顺序）堆叠数组构成一个新的数组
    velo2cam = np.vstack((velo2cam, np.array([0, 0, 0, 1.0])))

    # 获取图像尺寸
    im_shape = cam2cam["S_rect_02"][::-1].astype(np.int32)

    # 计算 velodyne->图像平面 的投影矩阵： 4 * 4
    R_cam2rect = np.eye(4)
    R_cam2rect[:3, :3] = cam2cam['R_rect_00'].reshape(3, 3)
    P_rect = cam2cam['P_rect_0'+str(cam)].reshape(3, 4)
    # np.dot()：矩阵乘法
    P_velo2im = np.dot(np.dot(P_rect, R_cam2rect), velo2cam)

    # 加载 velodyne 点并删除图像平面后面的所有点（近似值）
    # velodyne 数据的每一行是向前、向左、向上、反射率
    velo = load_velodyne_points(velo_filename)
    velo = velo[velo[:, 0] >= 0, :]

    # 将点投影到相机
    velo_pts_im = np.dot(P_velo2im, velo.T).T
    velo_pts_im[:, :2] = velo_pts_im[:, :2] / velo_pts_im[:, 2][..., np.newaxis]

    if vel_depth:
        velo_pts_im[:, 2] = velo[:, 0]

    # 检查是否在边界内
    # 使用 -1 得到与 KITTI matlab 代码完全相同的值
    velo_pts_im[:, 0] = np.round(velo_pts_im[:, 0]) - 1
    velo_pts_im[:, 1] = np.round(velo_pts_im[:, 1]) - 1
    val_inds = (velo_pts_im[:, 0] >= 0) & (velo_pts_im[:, 1] >= 0)
    val_inds = val_inds & (velo_pts_im[:, 0] < im_shape[1]) & (velo_pts_im[:, 1] < im_shape[0])
    velo_pts_im = velo_pts_im[val_inds, :]

    # 投影到图像
    depth = np.zeros((im_shape[:2]))
    depth[velo_pts_im[:, 1].astype(np.int), velo_pts_im[:, 0].astype(np.int)] = velo_pts_im[:, 2]

    # 找到重复点并选择最接近的深度
    inds = sub2ind(depth.shape, velo_pts_im[:, 1], velo_pts_im[:, 0])
    dupe_inds = [item for item, count in Counter(inds).items() if count > 1]
    for dd in dupe_inds:
        pts = np.where(inds == dd)[0]
        x_loc = int(velo_pts_im[pts[0], 0])
        y_loc = int(velo_pts_im[pts[0], 1])
        depth[y_loc, x_loc] = velo_pts_im[pts, 2].min()
    depth[depth < 0] = 0

    return depth
