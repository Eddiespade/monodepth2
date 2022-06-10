# Monodepth2
（用于学习用途的Monodepth2个人注解版）

训练和测试深度估计模型的参考 PyTorch 实现

> **Digging into Self-Supervised Monocular Depth Prediction**
>
> [Clément Godard](http://www0.cs.ucl.ac.uk/staff/C.Godard/), [Oisin Mac Aodha](http://vision.caltech.edu/~macaodha/), [Michael Firman](http://www.michaelfirman.co.uk) and [Gabriel J. Brostow](http://www0.cs.ucl.ac.uk/staff/g.brostow/)
>
> [ICCV 2019 (arXiv pdf)](https://arxiv.org/abs/1806.01260)

<p align="center">
  <img src="assets/teaser.gif" alt="example input output gif" width="600" />
</p>

此代码用于非商业用途； 有关条款，请参阅 [许可文件](LICENSE)。


如果您发现我们的工作对您的研究有用，请考虑引用我们的论文：

```
@article{monodepth2,
  title     = {Digging into Self-Supervised Monocular Depth Prediction},
  author    = {Cl{\'{e}}ment Godard and
               Oisin {Mac Aodha} and
               Michael Firman and
               Gabriel J. Brostow},
  booktitle = {The International Conference on Computer Vision (ICCV)},
  month = {October},
year = {2019}
}
```



## ⚙️ 设置

假设一个新的 [Anaconda](https://www.anaconda.com/download/) 发行版，您可以安装依赖项：
```shell
conda install pytorch=0.4.1 torchvision=0.2.1 -c pytorch
pip install tensorboardX==1.4
conda install opencv=3.3.1   # 只需要评估
```

我们使用 PyTorch 0.4.1、CUDA 9.1、Python 3.6.6 和 Ubuntu 18.04 进行了实验。
我们还成功地使用 PyTorch 1.0 训练了模型，并且我们的代码与 Python 2.7 兼容。 如果您使用 Python 3.7，安装 OpenCV 版本 3.3.1 可能会遇到问题，我们建议使用 Python 3.6.6 `conda create -n monodepth2 python=3.6.6 anaconda` 创建虚拟环境。

<!-- We recommend using a [conda environment](https://conda.io/docs/user-guide/tasks/manage-environments.html) to avoid dependency conflicts.

We also recommend using `pillow-simd` instead of `pillow` for faster image preprocessing in the dataloaders. -->


## 🖼️ 单个图像的预测

您可以使用以下方法预测单个图像的缩放视差：

```shell
python test_simple.py --image_path assets/test_image.jpg --model_name mono+stereo_640x192
```

或者，如果您使用的是立体训练模型，则可以使用

```shell
python test_simple.py --image_path assets/test_image.jpg --model_name mono+stereo_640x192 --pred_metric_depth
```

在第一次运行时，这些命令中的任何一个都会将 `mono+stereo_640x192` 预训练模型 (99MB) 下载到 `models/` 文件夹中。
我们为 `--model_name` 提供以下选项：

| `--model_name`          | Training modality | Imagenet pretrained? | Model resolution  | KITTI abs. rel. error |  delta < 1.25  |
|-------------------------|-------------------|--------------------------|-----------------|------|----------------|
| [`mono_640x192`](https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_640x192.zip)          | Mono              | Yes | 640 x 192                | 0.115                 | 0.877          |
| [`stereo_640x192`](https://storage.googleapis.com/niantic-lon-static/research/monodepth2/stereo_640x192.zip)        | Stereo            | Yes | 640 x 192                | 0.109                 | 0.864          |
| [`mono+stereo_640x192`](https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_640x192.zip)   | Mono + Stereo     | Yes | 640 x 192                | 0.106                 | 0.874          |
| [`mono_1024x320`](https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_1024x320.zip)         | Mono              | Yes | 1024 x 320               | 0.115                 | 0.879          |
| [`stereo_1024x320`](https://storage.googleapis.com/niantic-lon-static/research/monodepth2/stereo_1024x320.zip)       | Stereo            | Yes | 1024 x 320               | 0.107                 | 0.874          |
| [`mono+stereo_1024x320`](https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_1024x320.zip)  | Mono + Stereo     | Yes | 1024 x 320               | 0.106                 | 0.876          |
| [`mono_no_pt_640x192`](https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_no_pt_640x192.zip)          | Mono              | No | 640 x 192                | 0.132                 | 0.845          |
| [`stereo_no_pt_640x192`](https://storage.googleapis.com/niantic-lon-static/research/monodepth2/stereo_no_pt_640x192.zip)        | Stereo            | No | 640 x 192                | 0.130                 | 0.831          |
| [`mono+stereo_no_pt_640x192`](https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_no_pt_640x192.zip)   | Mono + Stereo     | No | 640 x 192                | 0.127                 | 0.836          |

您还可以下载使用 [monoocular](https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_odom_640x192.zip) 和 [mono+stereo](https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_odom_640x192.zip) 训练方式。

最后，我们提供了使用 [ImageNet 预训练权重](https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_resnet50_640x192.zip) 和 [从头开始训练](https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_resnet50_no_pt_640x192.zip)。
如果使用这些，请确保设置 `--num_layers 50`。

## 💾 KITTI 训练数据集

您可以通过运行以下命令下载整个 [原始 KITTI 数据集](http://www.cvlibs.net/datasets/kitti/raw_data.php)：
```shell
wget -i splits/kitti_archives_to_download.txt -P kitti_data/
```
然后解压
```shell
cd kitti_data
unzip "*.zip"
cd ..
```
**警告：** 它的文件大小约为 **175GB**，因此请确保您也有足够的空间来解压缩！

我们的默认设置要求您使用此命令将 png 图像转换为 jpeg，**这也会删除原始 KITTI `.png` 文件**：
```shell
find kitti_data/ -name '*.png' | parallel 'convert -quality 92 -sampling-factor 2x2,1x1,1x1 {.}.png {.}.jpg && rm {}'
```
**或**您可以跳过此转换步骤并通过在训练时添加标志 `--png` 从原始 png 文件进行训练，但会降低加载时间。

上面的转换命令创建了与我们的实验相匹配的图像，其中 KITTI `.png` 图像在 Ubuntu 16.04 上被转换为 `.jpg`，默认色度子采样为 `2x2,1x1,1x1`。
我们发现 Ubuntu 18.04 默认为 `2x2,2x2,2x2`，这给出了不同的结果，因此转换命令中的显式参数。

您还可以将 KITTI 数据集放置在您喜欢的任何位置，并在训练和评估期间使用 `--data_path` 标志指向它。

**数据集拆分**

训练/测试/验证拆分在 `splits/` 文件夹中定义。
默认情况下，代码将使用 KITTI 的标准 Eigen split 的 [Zhou's subset](https://github.com/tinghuiz/SfMLearner) 训练深度模型，该模型专为单目训练而设计。
您还可以使用新的 [benchmark split](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction) 或 [odometry split](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) 通过设置`--split`标志。

**自定义数据集**

您可以通过编写一个继承自“MonoDataset”的新数据加载器类在自定义单目或立体数据集上进行训练——参见“datasets/kitti_dataset.py”中的“KITTIDataset”类作为示例。

## ⏳ 训练

默认情况下，模型和tensorboard文件保存到 `~/tmp/<model_name>`。
这可以使用 `--log_dir` 标志进行更改。


**单目训练:**
```shell
python train.py --model_name mono_model
```

**立体训练:**

我们的代码默认使用 Zhou 的二次抽样 Eigen 训练数据。 对于纯立体训练，我们必须指定我们要使用完整的 Eigen 训练集——详见论文。
```shell
python train.py --model_name stereo_model \
  --frame_ids 0 --use_stereo --split eigen_full
```

**单目+立体训练:**
```shell
python train.py --model_name mono+stereo_model \
  --frame_ids 0 -1 1 --use_stereo
```


### GPUs

代码只能在单个 GPU 上运行。
您可以通过 `CUDA_VISIBLE_DEVICES` 环境变量指定要使用的 GPU：
```shell
CUDA_VISIBLE_DEVICES=2 python train.py --model_name mono_model
```

我们所有的实验都是在单个 NVIDIA Titan Xp 上进行的。

| Training modality | Approximate GPU memory  | Approximate training time   |
|-------------------|-------------------------|-----------------------------|
| Mono              | 9GB                     | 12 hours                    |
| Stereo            | 6GB                     | 8 hours                     |
| Mono + Stereo     | 11GB                    | 15 hours                    |



### 💽 微调预训练模型

将以下内容添加到训练命令以加载现有模型以进行微调：
```shell
python train.py --model_name finetuned_mono --load_weights_folder ~/tmp/mono_model/models/weights_19
```


### 🔧 其他训练选择

Run `python train.py -h` (or look at `options.py`) to see the range of other training options, such as learning rates and ablation settings.


## 📊 KITTI评估

要准备地面实况深度图，请运行：
```shell
python export_gt_depth.py --data_path kitti_data --split eigen
python export_gt_depth.py --data_path kitti_data --split eigen_benchmark
```
...假设您已将 KITTI 数据集放置在 `./kitti_data/` 的默认位置。

以下示例命令评估名为“mono_model”的模型的 epoch 19 权重：
```shell
python evaluate_depth.py --load_weights_folder ~/tmp/mono_model/models/weights_19/ --eval_mono
```
对于立体模型，您必须使用 `--eval_stereo` 标志（请参阅下面的注释）：
```shell
python evaluate_depth.py --load_weights_folder ~/tmp/stereo_model/models/weights_19/ --eval_stereo
```
如果您使用我们的代码训练您自己的模型，由于权重初始化和数据加载中的随机化，您可能会看到与发布结果略有不同。

可以设置一个附加参数 `--eval_split`。
`eval_split` 可能的三个不同值在这里解释：

| `--eval_split`        | 测试集大小 | 使用...训练的模型                                             | 描述                                                                                                                          |
|-----------------------|---------------|--------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------|
| **`eigen`**           | 697           | `--split eigen_zhou` (默认) or `--split eigen_full`      | 标准 Eigen 测试文件                                                                                                               |
| **`eigen_benchmark`** | 652           | `--split eigen_zhou` (默认) or `--split eigen_full` | 使用来自 [新 KITTI 深度基准测试](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction) 的改进的真实标签进行评估          |
| **`benchmark`**       | 500           | `--split benchmark`                                    | [新的 KITTI 深度基准测试](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction) 测试文件。 |

因为新的 KITTI 深度基准没有可用的基本事实，所以设置 --eval_split benchmark 时不会报告分数。
相反，一组“.png”图像将保存到磁盘上，准备上传到评估服务器。


**评估表面视差**

最后，您还可以使用 `evaluate_depth.py` 通过使用 `--ext_disp_to_eval` 标志来评估来自其他方法的原始差异（或反向深度）：

```shell
python evaluate_depth.py --ext_disp_to_eval ~/other_method_disp.npy
```


**📷📷 立体评估注意事项**

我们的立体模型以“0.1”单位的有效基线进行训练，而实际的 KITTI 立体装置的基线为“0.54m”。 这意味着必须应用“5.4”的比例进行评估。
此外，对于使用立体监督训练的模型，我们禁用中值缩放。
在评估时设置 `--eval_stereo` 标志将自动禁用中值缩放并将预测深度缩放 `5.4`。


**⤴️⤵️ 里程计评估**

我们包含用于评估由使用 --split odom --dataset kitti_odom --data_path /path/to/kitti/odometry/dataset 训练的模型预测的姿态的代码。

对于此评估，[KITTI odometry dataset](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) **(彩色, 65GB)** 和 **真实标签位姿** zip 文件必须 下载。
如上所述，我们假设 png 已转换为 jpg。

如果此数据已解压缩到文件夹“kitti_odom”，则可以使用以下方法评估模型：

```shell
python evaluate_pose.py --eval_split odom_9 --load_weights_folder ./odom_split.M/models/weights_29 --data_path kitti_odom/
python evaluate_pose.py --eval_split odom_10 --load_weights_folder ./odom_split.M/models/weights_29 --data_path kitti_odom/
```


## 📦 预训练权重结果

您可以从以下链接下载我们预先计算的差异预测：


| 训练方式          | 输入尺寸       | `.npy` 文件大小 | Eigen 视差                                                                                                             |
|---------------|------------|-------------|----------------------------------------------------------------------------------------------------------------------|
| Mono          | 640 x 192  | 343 MB      | [Download 🔗](https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_640x192_eigen.npy)          |
| Stereo        | 640 x 192  | 343 MB      | [Download 🔗](https://storage.googleapis.com/niantic-lon-static/research/monodepth2/stereo_640x192_eigen.npy)        |
| Mono + Stereo | 640 x 192  | 343 MB      | [Download 🔗](https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_640x192_eigen.npy) |
| Mono          | 1024 x 320 | 914 MB      | [Download 🔗](https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_1024x320_eigen.npy)         |
| Stereo        | 1024 x 320 | 914 MB      | [Download 🔗](https://storage.googleapis.com/niantic-lon-static/research/monodepth2/stereo_1024x320_eigen.npy)       |
| Mono + Stereo | 1024 x 320 | 914 MB      | [Download 🔗](https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_1024x320_eigen.npy) |



## 👩‍⚖️ 许可
版权所有 © Niantic, Inc. 2019。专利申请中。
版权所有。
有关条款，请参阅 [许可文件](LICENSE)。
