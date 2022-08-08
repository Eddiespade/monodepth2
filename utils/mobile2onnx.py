import torch
import torch.onnx
import os

import torch.nn as nn
import os
import cv2
import glob
import torch
import networks
import argparse
import numpy as np
import PIL.Image as pil

from layers import disp_to_depth
from torchvision import transforms


class FullModel(nn.Module):
    def __init__(self, encoder, decoder):
        super(FullModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        features = self.encoder(x)
        outputs = self.decoder(features)
        # print(outputs[("disp", 0)].shape)
        return outputs[("disp", 0)]


def pth_to_onnx(input, checkpoint, checkpoint1, onnx_path, input_names=['input'], output_names=['output'],
                device='cpu'):
    if not onnx_path.endswith('.onnx'):
        print('Warning! The onnx model name is not correct,\
              please give a name that ends with \'.onnx\'!')
        return 0
    # model = Model()
    device = torch.device("cuda:2")
    encoder_path = checkpoint
    depth_decoder_path = checkpoint1
    encoder = networks.ResnetEncoder(18, False)
    loaded_dict_enc = torch.load(encoder_path, map_location=device)

    # 提取模型训练时使用的图像的高度和宽度
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    print(feed_width, feed_height)

    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    # encoder = nn.DataParallel(encoder, device_ids=[2])
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()
    #
    # print("------ Loading pretrained decoder ------")
    depth_decoder = networks.DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))

    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    # depth_decoder = nn.DataParallel(depth_decoder, device_ids=[2])
    depth_decoder.load_state_dict(loaded_dict)
    depth_decoder.to(device)
    depth_decoder.eval()

    full_model = FullModel(encoder, depth_decoder)
    full_model = full_model.eval()

    torch.onnx.export(full_model, input, onnx_path, verbose=False, opset_version=11)  # 指定模型的输入，以及onnx的输出路径
    print("Exporting .pth model to onnx model has been successful!")


if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    checkpoint = './pth/encoder.pth'
    checkpoint1 = './pth/depth.pth'
    onnx_path = './monodepth.onnx'

    device = torch.device("cuda:2" if torch.cuda.is_available() else 'cpu')
    input = torch.randn(1, 3, 480, 640).to(device)
    # x = torch.randn((1, 3, 192, 640), device="cuda")

    pth_to_onnx(input, checkpoint, checkpoint1, onnx_path, device="")
