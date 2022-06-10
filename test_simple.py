# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import sys
import glob
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm

import torch
from torchvision import transforms, datasets

import networks
from layers import disp_to_depth
from utils import download_model_if_doesnt_exist
from evaluate_depth import STEREO_SCALE_FACTOR


def parse_args():
    parser = argparse.ArgumentParser(
        description='Simple testing funtion for Monodepthv2 models.')

    # 测试图像或图像文件夹的路径(必须！)
    parser.add_argument('--image_path', type=str,
                        help='path to a test image or folder of images', required=True)
    # 要使用的预训练模型的名称
    parser.add_argument('--model_name', type=str,
                        help='name of a pretrained model to use',
                        choices=[
                            "stereo1_640x192",
                            "mono_640x192",
                            "stereo_640x192",
                            "mono+stereo_640x192",
                            "mono_no_pt_640x192",
                            "stereo_no_pt_640x192",
                            "mono+stereo_no_pt_640x192",
                            "mono_1024x320",
                            "stereo_1024x320",
                            "mono+stereo_1024x320"])
    # 在文件夹中搜索的图像扩展名。默认为 jpg
    parser.add_argument('--ext', type=str,
                        help='image extension to search for in folder', default="jpg")
    # 如果设置这一选项，则使用cpu测试
    parser.add_argument("--no_cuda",
                        help='if set, disables CUDA',
                        action='store_true')
    # 如果设置，则预测度量深度而不是视差。（这只对立体训练的 KITTI 模型有意义）
    parser.add_argument("--pred_metric_depth",
                        help='if set, predicts metric depth instead of disparity. (This only '
                             'makes sense for stereo-trained KITTI models).',
                        action='store_true')

    return parser.parse_args()


def test_simple(args):
    """
    预测单个图像或图像文件夹的功能
    """
    assert args.model_name is not None, \
        "You must specify the --model_name parameter; see README.md for an example"

    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if args.pred_metric_depth and "stereo" not in args.model_name:
        print("Warning: The --pred_metric_depth flag only makes sense for stereo-trained KITTI "
              "models. For mono-trained models, output depths will not in metric space.")

    # 如果模型不存在，则尝试从作者给定的地址下载模型到本地
    download_model_if_doesnt_exist(args.model_name)
    model_path = os.path.join("models", args.model_name)
    print("-> Loading model from ", model_path)
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")

    """ 加载预训练模型的 encoder层；注意 18 需要根据实际进行修改 """
    print("   Loading pretrained encoder")
    encoder = networks.ResnetEncoder(18, False)
    loaded_dict_enc = torch.load(encoder_path, map_location=device)

    # 提取该模型训练时使用的图像的高度和宽度
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()          # 开启验证模式

    """ 加载预训练的 decoder层 """
    print("   Loading pretrained decoder")
    depth_decoder = networks.DepthDecoder(
        num_ch_enc=encoder.num_ch_enc, scales=range(4))

    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)
    depth_decoder.to(device)
    depth_decoder.eval()    # 开启验证模式

    # 查找输入图像
    if os.path.isfile(args.image_path):
        # 仅在单个图像上测试
        paths = [args.image_path]
        # os.path.dirname(file): 功能 - 去掉file的文件名，返回其上一级目录
        output_directory = os.path.dirname(args.image_path)
    elif os.path.isdir(args.image_path):
        # 在文件夹中搜索图像；glob.glob()：其功能是查找符合特定规则的文件路径，返回所有符合要求的列表
        paths = glob.glob(os.path.join(args.image_path, '*.{}'.format(args.ext)))
        output_directory = args.image_path
    else:
        raise Exception("Can not find args.image_path: {}".format(args.image_path))

    print("-> Predicting on {:d} test images".format(len(paths)))

    # 依次预测每个图像
    with torch.no_grad():
        for idx, image_path in enumerate(paths):

            if image_path.endswith("_disp.jpg"):
                # 如果已经是视差图了，就不要试图再去预测视差图像的视差！
                continue

            # 加载图像和预处理
            input_image = pil.open(image_path).convert('RGB')           # 打开图片并转换为RGB图像
            original_width, original_height = input_image.size          # 获取原始图片的宽高尺寸
            input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)    # resize到需要的尺寸；LANCZOS：高质量下采样滤波器
            input_image = transforms.ToTensor()(input_image).unsqueeze(0)               # numpy转换为tensor

            # 开始预测
            input_image = input_image.to(device)
            features = encoder(input_image)
            outputs = depth_decoder(features)

            disp = outputs[("disp", 0)]
            # 将预测输出的视差图 利用双线性采样到原始的图像尺寸上
            disp_resized = torch.nn.functional.interpolate(
                disp, (original_height, original_width), mode="bilinear", align_corners=False)

            # 保存numpy文件
            # os.path.splitext(“文件路径”)    分离文件名与扩展名；默认返回(fname, fextension)元组，可做分片操作
            output_name = os.path.splitext(os.path.basename(image_path))[0]
            # 将网络的 sigmoid 输出转换为深度预测
            scaled_disp, depth = disp_to_depth(disp, 0.1, 100)
            if args.pred_metric_depth:      # 预测深度
                name_dest_npy = os.path.join(output_directory, "{}_depth.npy".format(output_name))
                # STEREO_SCALE_FACTOR：深度比例大小，文中取5.4
                metric_depth = STEREO_SCALE_FACTOR * depth.cpu().numpy()
                # ------------------------------------------------------------------------------------------------------
                # 语法：np.save(file, arr, allow_pickle=True, fix_imports=True)
                # 作用：以“.npy”格式将数组保存到二进制文件中。
                # 参数：
                #   file 要保存的文件名称，需指定文件保存路径，如果未设置，保存到默认路径。其文件拓展名为.npy
                #   arr 为需要保存的数组，也即把数组arr保存至名称为file的文件中。
                # ------------------------------------------------------------------------------------------------------
                np.save(name_dest_npy, metric_depth)
            else:                           # 预测视差
                name_dest_npy = os.path.join(output_directory, "{}_disp.npy".format(output_name))
                np.save(name_dest_npy, scaled_disp.cpu().numpy())

            # 保存彩色映射深度图像
            disp_resized_np = disp_resized.squeeze().cpu().numpy()                      # 输出的视差图的转到 numpy上
            # ----------------------------------------------------------------------------------------------------------
            # 语法：np.percentile(a, q, axis=None, out=None, overwrite_input=False, interpolation='linear', keepdims=False)
            # 作用：找到一组数的分位数值，如四分位数等(具体什么位置根据自己定义)
            # 参数：
            #   a   : array，用来算分位数的对象，可以是多维的数组
            #   q   : 介于0-100的float，用来计算是几分位的参数，如四分之一位就是25，如要算两个位置的数就(25,75)
            #   axis    : 坐标轴的方向，一维的就不用考虑了，多维的就用这个调整计算的维度方向，取值范围0/1
            #   out     : 输出数据的存放对象，参数要与预期输出有相同的形状和缓冲区长度
            #   overwrite_input : bool，默认False，为True时及计算直接在数组内存计算，计算后原数组无法保存
            #   interpolation   : 取值范围{'linear', 'lower', 'higher', 'midpoint', 'nearest'} 默认liner，
            #                   比如取中位数，但是中位数有两个数字6和7，选不同参数来调整输出
            #   keepdims : bool,默认False,为真时取中位数的那个轴将保留在结果中
            # ----------------------------------------------------------------------------------------------------------
            vmax = np.percentile(disp_resized_np, 95)    # 找到该视差图的95分位数
            # 一个类，当被调用时，它将数据线性地标准化为 [0.0, 1.0] 间隔。
            normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
            # 将标量数据映射到 RGBA 的 mixin 类；ScalarMappable 在从给定颜色图中返回 RGBA 颜色之前应用数据规范化[0,1]
            mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
            im = pil.fromarray(colormapped_im)

            name_dest_im = os.path.join(output_directory, "{}_disp.jpeg".format(output_name))
            im.save(name_dest_im)

            print("   Processed {:d} of {:d} images - saved predictions to:".format(
                idx + 1, len(paths)))
            print("   - {}".format(name_dest_im))
            print("   - {}".format(name_dest_npy))

    print('-> Done!')


if __name__ == '__main__':
    args = parse_args()
    test_simple(args)
