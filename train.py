from __future__ import absolute_import, division, print_function

from trainer import Trainer
from options import MonodepthOptions

# 命令行传参
options = MonodepthOptions()
# 解析参数
opts = options.parse()


if __name__ == "__main__":
    trainer = Trainer(opts)
    trainer.train()
