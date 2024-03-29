from __future__ import absolute_import, division, print_function

import os
import cv2
import numpy as np

import torch
from torch.utils.data import DataLoader

from layers import disp_to_depth
from utils import readlines
from options import MonodepthOptions
import datasets
import networks
import PIL.Image as pil

cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)


splits_dir = os.path.join(os.path.dirname(__file__), "splits")

# 用立体监督训练的模型用名义训练
# 0.1 个单位的基线。 KITTI 钻机的基线为 54 厘米。 所以，
# 为了将我们的立体预测转换为真实世界的比例，我们将深度乘以 5.4。
STEREO_SCALE_FACTOR = 5.4


def compute_errors(gt, pred):
    """
    计算预测和真实深度之间的误差度量
    """
    # 阈值 始终大于1
    thresh = np.maximum((gt / pred), (pred / gt))
    # ai 表示不同阈值下所占的比率
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    # rmse：均方根误差
    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    # rmse_log: log后的均方根误差
    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    # abs_rel： 相对的绝对值误差 ( / 真实标签)
    abs_rel = np.mean(np.abs(gt - pred) / gt)
    # sq_rel： 相对的平方误差 ( / 真实标签)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def batch_post_process_disparity(l_disp, r_disp):
    """
    应用 Monodepthv1 中介绍的视差后处理方法
    """
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


def evaluate(opt):
    """
    使用指定的测试集评估预训练模型
    """
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80
    my_idx = 0

    assert sum((opt.eval_mono, opt.eval_stereo)) == 1, \
        "Please choose mono or stereo evaluation by setting either --eval_mono or --eval_stereo"

    if opt.ext_disp_to_eval is None:

        opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)

        assert os.path.isdir(opt.load_weights_folder), \
            "Cannot find a folder at {}".format(opt.load_weights_folder)

        print("-> Loading weights from {}".format(opt.load_weights_folder))

        # 测试数据集路径
        filenames = readlines(os.path.join(splits_dir, opt.eval_split, "test_files.txt"))
        encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
        decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")

        encoder_dict = torch.load(encoder_path)

        dataset = datasets.KITTIRAWDataset(opt.data_path, filenames,
                                           encoder_dict['height'], encoder_dict['width'],
                                           [0], 4, is_train=False)
        dataloader = DataLoader(dataset, 16, shuffle=False, num_workers=opt.num_workers,
                                pin_memory=True, drop_last=False)

        encoder = networks.ResnetEncoder(opt.num_layers, False)
        depth_decoder = networks.DepthDecoder(encoder.num_ch_enc)

        model_dict = encoder.state_dict()
        encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
        depth_decoder.load_state_dict(torch.load(decoder_path))

        encoder.cuda()
        encoder.eval()
        depth_decoder.cuda()
        depth_decoder.eval()

        # 存储预测的视差图
        pred_disps = []

        print("-> Computing predictions with size {}x{}".format(
            encoder_dict['width'], encoder_dict['height']))

        import time
        start = time.time()
        with torch.no_grad():
            for data in dataloader:
                input_color = data[("color", 0, 0)].cuda()

                if opt.post_process:
                    # 后处理结果要求每张图像有两次前向传递
                    input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)

                output = depth_decoder(encoder(input_color))

                pred_disp, _ = disp_to_depth(output[("disp", 0)], opt.min_depth, opt.max_depth)
                pred_disp = pred_disp.cpu()[:, 0].numpy()

                # 后处理
                if opt.post_process:
                    N = pred_disp.shape[0] // 2
                    pred_disp = batch_post_process_disparity(pred_disp[:N], pred_disp[N:, :, ::-1])

                pred_disps.append(pred_disp)
        end = time.time()
        print("running time is ",  float(end - start) * 1000.0, "ms")

        # np.concatenate 对array进行拼接的函数
        pred_disps = np.concatenate(pred_disps)

    else:
        # 如果存在.npy文件，则从文件加载预测
        print("-> Loading predictions from {}".format(opt.ext_disp_to_eval))
        pred_disps = np.load(opt.ext_disp_to_eval)

        if opt.eval_eigen_to_benchmark:
            eigen_to_benchmark_ids = np.load(
                os.path.join(splits_dir, "benchmark", "eigen_to_benchmark_ids.npy"))

            pred_disps = pred_disps[eigen_to_benchmark_ids]

    # 如果设置后，则保存预测的差异；
    if opt.save_pred_disps:
        output_path = os.path.join(
            opt.load_weights_folder, "disps_{}_split.npy".format(opt.eval_split))
        print("-> Saving predicted disparities to ", output_path)
        np.save(output_path, pred_disps)

    # 如果设置，则不评估
    if opt.no_eval:
        print("-> Evaluation disabled. Done.")
        quit()

    # KITTI benchmark没有可用的groud truth，因此不进行评估。
    elif opt.eval_split == 'benchmark':
        save_dir = os.path.join(opt.load_weights_folder, "benchmark_predictions")
        print("-> Saving out benchmark predictions to {}".format(save_dir))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for idx in range(len(pred_disps)):
            disp_resized = cv2.resize(pred_disps[idx], (1216, 352))
            depth = STEREO_SCALE_FACTOR / disp_resized
            # np.clip是一个截取函数，用于截取数组中小于或者大于某值的部分，并使得被截取部分等于固定值。
            depth = np.clip(depth, 0, 80)
            depth = np.uint16(depth * 256)
            save_path = os.path.join(save_dir, "{:010d}.png".format(idx))
            cv2.imwrite(save_path, depth)

        print("-> No ground truth is available for the KITTI benchmark, so not evaluating. Done.")
        quit()

    # 加载真实深度信息
    gt_path = os.path.join(splits_dir, opt.eval_split, "gt_depths.npz")
    # -----------------------------------------------------------------------------------------------
    # 用法：numpy.load(file, mmap_mode=None, allow_pickle=True, fix_imports=True,encoding=’ASCII’)
    # 参数：
    #     file ：file-like对象，字符串或pathlib.Path。要读取的文件。 File-like对象必须支持seek()和read()方法。
    #     mmap_mode :如果不为None，则使用给定模式memory-map文件(有关详细信息，请参见numpy.memmap
    #     模式说明)。
    #     allow_pickle :允许加载存储在npy文件中的腌制对象数组。
    #     fix_imports :仅在在Python 3上加载Python 2生成的腌制文件时有用，该文件包括包含对象数组的npy /npz文件。
    #     encoding :仅当在Python 3中加载Python 2生成的腌制文件时有用，该文件包含包含对象数组的npy /npz文件。
    #     Returns :数据存储在文件中。对于.npz文件，必须关闭NpzFile类的返回实例，以避免泄漏文件描述符。
    # -----------------------------------------------------------------------------------------------
    gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]

    # ======================================= 开始评估 ===========================================
    print("-> Evaluating")
    # 单目评估还是双目评估
    if opt.eval_stereo:
        print("   Stereo evaluation - "
              "disabling median scaling, scaling by {}".format(STEREO_SCALE_FACTOR))
        opt.disable_median_scaling = True
        opt.pred_depth_scale_factor = STEREO_SCALE_FACTOR
    else:
        print("   Mono evaluation - using median scaling")

    errors = []     # 保存真实深度和预测深度的差距
    ratios = []     # 保存真实深度/预测深度的比率

    for i in range(pred_disps.shape[0]):

        gt_depth = gt_depths[i]
        gt_height, gt_width = gt_depth.shape[:2]

        pred_disp = pred_disps[i]
        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        # ----------------------------------- 添加保存图片 ----------------------------------------
        pil.fromarray(pred_disp).convert('RGB').save("./assets/predict_disp/{}.png".format(my_idx))
        pred_depth = 1 / pred_disp
        pil.fromarray(pred_depth).convert('RGB').save("./assets/predict_depth/{}.png".format(my_idx))
        my_idx += 1

        if opt.eval_split == "eigen":
            # np.logical_and(): 数组的矩阵的逻辑操作-与
            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

            crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                             0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
            crop_mask = np.zeros(mask.shape)
            crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
            # mask 仅在 真实深度在最大和最小值之间 且 在规定的区间 才为1
            mask = np.logical_and(mask, crop_mask)

        else:
            # 在此情况下，仅需真实深度有效 即 > 0时才 为 True
            mask = gt_depth > 0

        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]

        pred_depth *= opt.pred_depth_scale_factor
        # 对预测深度进行中值缩放
        if not opt.disable_median_scaling:
            ratio = np.median(gt_depth) / np.median(pred_depth)
            ratios.append(ratio)
            pred_depth *= ratio

        # 将预测的深度信息 限制在 [MIN_DEPTH, MAX_DEPTH]区间
        pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
        pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

        # 计算损失并保存
        errors.append(compute_errors(gt_depth, pred_depth))

    # 打印中指缩放的相关信息
    if not opt.disable_median_scaling:
        ratios = np.array(ratios)
        # np.std()函数被用来：计算沿指定轴的标准差。
        med = np.median(ratios)
        print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))

    # 打印总体的每一项平均损失
    mean_errors = np.array(errors).mean(0)

    print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
    print("\n-> Done!")


if __name__ == "__main__":
    options = MonodepthOptions().parse()
    options.load_weights_folder = "~/dxl/test-dl/monodepth2/models/mono_640x192"
    options.eval_mono = True
    options.png = True
    evaluate(options)
