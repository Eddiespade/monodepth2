"""
此代码用于测试自己数据集性能
数据集形式：
    待预测的图片（文件夹） ——————> 一个文件夹下面有多个图片
    真实视差标签（文件夹） ——————> 一个文件夹下面有多个图片
    要求：标签文件名和带预测图片文件名的后 10位能够匹配上；如image：_left_image003171.png， disp：_disp003171.png  能匹配的字段为：003171.png

基本思路：
    1. 加载训练好的网络模型 （模型需要提前具备）
    2. 将待预测的图片通过网络预测并保存输出的视差图
    3. 加载保存的视差图和真实视差标签进行模型性能评估
    4. 转换为深度图进行模型性能评估
"""

import os
import shutil

import cv2
import glob
import torch
import networks
import argparse
import numpy as np
import PIL.Image as pil

from layers import disp_to_depth
from torchvision import transforms

from utils import readlines

STEREO_SCALE_FACTOR = 5.2


def parse():
    parser = argparse.ArgumentParser(description='evaluation of our dataset')
    parser.add_argument('--data_path', default='../mono_depth_v2_data',
                        type=str, help='the folder of datasets')
    parser.add_argument('--model_folder', default='runs/finetuned/models/weights_19',
                        type=str, help='the folder of pre-model')
    parser.add_argument('--test_txt', default='splits/eigen_007/test_files.txt',
                        type=str, help='the folder of image')
    parser.add_argument('--image_folder', default='../data/image/',
                        type=str, help='the folder of predict disp')
    parser.add_argument('--disp_folder', default='../data/predict_disp/',
                        type=str, help='the folder of predict disp')
    parser.add_argument('--depth_folder', default='../data/predict_depth/',
                        type=str, help='the folder of predict depth')
    parser.add_argument('--no_cuda', action="store_true", help='if use, disables CUDA')
    parser.add_argument('--ext', default='.jpg',
                        type=str, help='image extension to search for in folder')

    opt = parser.parse_args()
    return opt


def get_image_path(opt, folder, frame_index, side):
    f_str = "{:010d}{}".format(frame_index, opt.ext)
    side_map = {"2": 2, "3": 3, "l": 2, "r": 3}
    image_path = os.path.join(
        opt.data_path, folder, "image_0{}/data".format(side_map[side]), f_str)
    return image_path


def get_predict_disp(opt):
    # ------------------------------------------ 1. 加载训练好的网络模型 ---------------------------------------------------
    assert opt.model_folder is not None, \
        "You must specify the --model_folder parameter"

    if torch.cuda.is_available() and not opt.no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(opt.no_cuda)
    print(device)

    print("Loading model from： ", opt.model_folder)
    encoder_path = os.path.join(opt.model_folder, "encoder.pth")
    depth_decoder_path = os.path.join(opt.model_folder, "depth.pth")

    print("------ Loading pretrained encoder ------")
    encoder = networks.ResnetEncoder(18, False)
    loaded_dict_enc = torch.load(encoder_path, map_location=device)

    # 提取模型训练时使用的图像的高度和宽度
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']

    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()

    print("------ Loading pretrained decoder ------")
    depth_decoder = networks.DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))

    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)
    depth_decoder.to(device)
    depth_decoder.eval()

    # ---------------------------------- 2. 将待预测的图片通过网络预测并保存输出的视差图 ---------------------------------------
    print("----------- predict images -------------")
    # 找到所有的带预测图片
    test_filenames = readlines(opt.test_txt)
    print("-> Predicting on {:d} test images".format(len(test_filenames)))

    # 依次预测每张图片
    with torch.no_grad():
        for idx in range(len(test_filenames)):
            line = test_filenames[idx].split()
            folder = line[0]
            frame_index = int(line[1])
            side = line[2]
            if not os.path.exists(opt.image_folder):
                os.makedirs(opt.image_folder)
            image_path = get_image_path(opt, folder, frame_index, side)
            name = os.path.basename(image_path)
            shutil.copyfile(image_path, opt.image_folder + name)

            # 加载图片并预处理
            input_image = pil.open(image_path).convert('RGB')
            original_width, original_height = input_image.size
            input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
            input_image = transforms.ToTensor()(input_image).unsqueeze(0)

            # 对图片进行预测
            input_image = input_image.to(device)
            input_image = (input_image - 0.45) / 0.225
            features = encoder(input_image)
            outputs = depth_decoder(features)

            # 获取模型预测的视差图
            disp = torch.nn.functional.interpolate(
                outputs[("disp", 0)], (original_height, original_width), mode="bilinear", align_corners=False)
            scaled_disp, depth = disp_to_depth(disp, 0.1, 100)

            # 保存视差图与深度图
            if not os.path.exists(opt.depth_folder):
                os.makedirs(opt.depth_folder)
            if not os.path.exists(opt.disp_folder):
                os.makedirs(opt.disp_folder)
            output_name = os.path.splitext(os.path.basename(image_path)[-10:])[0]
            scaled_disp = scaled_disp.squeeze().cpu().numpy()
            scaled_disp = 10 * scaled_disp
            depth = depth.squeeze().cpu().numpy()
            depth = STEREO_SCALE_FACTOR * depth
            cv2.imwrite(opt.disp_folder + "_disp{}.png".format(output_name), scaled_disp)
            cv2.imwrite(opt.depth_folder + "_depth{}.png".format(output_name), depth)

    print('-> Done!')
    pass


if __name__ == '__main__':
    opt = parse()
    get_predict_disp(opt)
