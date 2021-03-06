
from __future__ import absolute_import, division, print_function
import os
import hashlib
import zipfile
from six.moves import urllib


def readlines(filename):
    """
    读取文本文件中的所有行并以列表形式返回
    """
    with open(filename, 'r') as f:
        # f.readlines()后面有加\n --- f.read().splitlines()没有\n
        lines = f.read().splitlines()
    return lines


def normalize_image(x):
    """
    归一化：调整图像像素的大小 到 [0, 1]范围
    """
    ma = float(x.max().cpu().data)      # 获取 x 的最大值
    mi = float(x.min().cpu().data)      # 获取 x 的最小值
    d = ma - mi if ma != mi else 1e5    # 防止除数为 0
    return (x - mi) / d


def sec_to_hm(t):
    """
    以秒为单位的时间转换为以小时、分钟和秒为单位的时间；例如 10239 -> (2, 50, 39)
    """
    t = int(t)
    s = t % 60
    t //= 60
    m = t % 60
    t //= 60
    return t, m, s


def sec_to_hm_str(t):
    """
    以秒为单位将时间转换为字符串形式；例如 10239 -> '02h50m39s'
    """
    h, m, s = sec_to_hm(t)
    return "{:02d}h{:02d}m{:02d}s".format(h, m, s)


def download_model_if_doesnt_exist(model_name):
    """
    如果预训练的 kitti 模型不存在，下载并解压
    """
    # values are tuples of (<google cloud URL>, <md5 checksum>)
    download_paths = {
        "mono_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_640x192.zip",
             "a964b8356e08a02d009609d9e3928f7c"),
        "stereo_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/stereo_640x192.zip",
             "3dfb76bcff0786e4ec07ac00f658dd07"),
        "mono+stereo_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_640x192.zip",
             "c024d69012485ed05d7eaa9617a96b81"),
        "mono_no_pt_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_no_pt_640x192.zip",
             "9c2f071e35027c895a4728358ffc913a"),
        "stereo_no_pt_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/stereo_no_pt_640x192.zip",
             "41ec2de112905f85541ac33a854742d1"),
        "mono+stereo_no_pt_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_no_pt_640x192.zip",
             "46c3b824f541d143a45c37df65fbab0a"),
        "mono_1024x320":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_1024x320.zip",
             "0ab0766efdfeea89a0d9ea8ba90e1e63"),
        "stereo_1024x320":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/stereo_1024x320.zip",
             "afc2f2126d70cf3fdf26b550898b501a"),
        "mono+stereo_1024x320":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_1024x320.zip",
             "cdc5fc9b23513c07d5b19235d9ef08f7"),
        }

    # 这里限制了模型的权重目录只能在本项目的 models 下
    if not os.path.exists("models"):
        os.makedirs("models")

    model_path = os.path.join("models", model_name)

    # 检查md5
    def check_file_matches_md5(checksum, fpath):
        if not os.path.exists(fpath):
            return False
        with open(fpath, 'rb') as f:
            current_md5checksum = hashlib.md5(f.read()).hexdigest()
        return current_md5checksum == checksum

    # 检查是否已经下载了模型；如果不存在，则按给定的地址下载到本地
    if not os.path.exists(os.path.join(model_path, "encoder.pth")):

        # model_url：保存的下载url路径；    required_md5checksum：MD5校验
        model_url, required_md5checksum = download_paths[model_name]

        if not check_file_matches_md5(required_md5checksum, model_path + ".zip"):
            print("-> Downloading pretrained model to {}".format(model_path + ".zip"))
            # -----------------------------------------------------------------------------------
            # 语法： urllib.request.urlretrieve(url, filename=None, reporthook=None, data=None)
            # 用途： 将URL表示的网络对象复制到本地文件。
            # url：          外部或者本地url
            # filename：     指定了保存到本地的路径（如果未指定该参数，urllib会生成一个临时文件来保存数据）；
            # reporthook：   是一个回调函数，当连接上服务器、以及相应的数据块传输完毕的时候会触发该回调。
            #               我们可以利用这个回调函数来显示当前的下载进度。
            # data：         指post到服务器的数据。该方法返回一个包含两个元素的元组(filename, headers)，
            #               filename表示保存到本地的路径，header表示服务器的响应头。
            # -------------------------------------------------------------------------------------------------
            urllib.request.urlretrieve(model_url, model_path + ".zip")

        # 再次检查一遍（上述已经下载了）；如果下载错误，则退出
        if not check_file_matches_md5(required_md5checksum, model_path + ".zip"):
            print("   Failed to download a file which matches the checksum - quitting")
            quit()

        # 下载正确则解压缩
        print("   Unzipping model...")
        with zipfile.ZipFile(model_path + ".zip", 'r') as f:
            # ZipFile对象的extractall(file_path): 解压ZIP文件，压缩包中的文件都放到 file_path 目录中。
            f.extractall(model_path)

        print("   Model unzipped to {}".format(model_path))
