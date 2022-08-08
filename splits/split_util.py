"""
此脚本用于划分自己的数据集： 数据集为保存在一个文件夹下
- 该文件夹内部样式：
2011_09_30
    ├── 2011_09_30_drive_0020_sync
    │   ├── image_02
    │   │   └── data                    存放左目图像
    │   ├── image_03
    │   │   └── data                    存放右目图像
    │   └── velodyne_points             (可选)雷达点云数据
    │       ├── data
    │       ├── timestamps_end.txt
    │       ├── timestamps_start.txt
    │       └── timestamps.txt
    ├── calib_cam_to_cam.txt            相机参数
    ├── calib_imu_to_velo.txt           相机参数
    └── calib_velo_to_cam.txt           相机参数

根据划分到每个 {}_files.txt 文件中：
每一行为：            文件夹 帧数 左目/右目
    如：2011_09_30/2011_09_30_drive_0020_sync 1 l

步骤：
    1. 找到每个需要划分的集合
    2. 根据集合中的 index 或者 打乱集合搞
"""
import os
import glob
import random

train_per = 0.7
valid_per = 0.2
test_per = 0.1

if __name__ == '__main__':
    data_path = "../../mono_depth_v2_data/"
    output_path = "./eigen_007/"
    if os.path.exists(output_path):
        for file in glob.glob(output_path + "*"):
            os.remove(file)
    else:
        os.mkdir(output_path)

    train_list, val_list, test_list = [], [], []
    for sDir in os.listdir(data_path):
        imgs_list_l = glob.glob(os.path.join(data_path, sDir) + '/image_02/data/*.png')
        imgs_list_r = glob.glob(os.path.join(data_path, sDir) + '/image_03/data/*.png')
        imgs_list = imgs_list_l + imgs_list_r
        imgs_num = len(imgs_list)
        imgs_num_l = len(imgs_list_l)
        imgs_num_r = len(imgs_list_r)
        random.seed(666)
        random.shuffle(imgs_list)

        test_point = int(imgs_num * test_per)
        train_point = int(imgs_num * (test_per + train_per))

        for i in range(imgs_num):
            filename = "val_files.txt"
            if i < test_point or imgs_list[i].endswith("0000000000.png") or imgs_list[i].endswith(
                    "{:010d}.png".format(imgs_num_l - 1)):
                filename = "test_files.txt"
            elif i < train_point:
                filename = "train_files.txt"
            curlist = imgs_list[i].split("/")[-1].split(".")[0]
            print(imgs_list[i])
            print(curlist)
            num = int(curlist)
            # 写入文件。如果filename不存在会自动创建， 'w'表示写数据，写之前会清空文件中的原有数据！
            with open(output_path + filename, 'a') as f:
                if "image_02" in imgs_list[i]:
                    f.write(sDir + " " + str(num) + " l\n")
                else:
                    f.write(sDir + " " + str(num) + " r\n")
