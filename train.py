from __future__ import absolute_import, division, print_function

from trainer import Trainer
from options import MonodepthOptions

# 命令行传参
options = MonodepthOptions()
# 解析参数
opts = options.parse()


if __name__ == "__main__":
    opts.model_name = "mono_model"
    opts.load_weight_folder = "models/mono_640x192"
    # -----------------------双目+单目---------
    # opts.use_stereo = True
    # opts.frame_ids = [0, -1, 1]
    trainer = Trainer(opts)
    trainer.train()
