# autobuilder

`auto_train.py` 提供了一个命令行入口，可以根据任务类型自动组装 PlugNPlay 仓库中的模块并启动训练。

## 快速开始

```bash
pip install torch torchvision
python auto_train.py --task object_detection --target-loss 0.6
```

默认使用仓库中的 AFPN 检测模块与 ResNet-50 主干拼接成 Faster R-CNN 检测器，并在脚本构造的简易合成数据集上训练，直到验证损失达到指定阈值或超过最大迭代轮数。

可选参数可查看 `python auto_train.py --help`。
