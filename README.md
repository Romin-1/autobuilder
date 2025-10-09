# autobuilder

`auto_train.py` 提供了一个命令行入口，可以根据任务类型自动组装 PlugNPlay 仓库中的模块并启动训练。

## 快速开始

```bash
pip install torch torchvision
python auto_train.py --task object_detection --target-loss 0.6
```

默认使用仓库中的 AFPN 检测模块与 ResNet-50 主干拼接成 Faster R-CNN 检测器，并在脚本构造的简易合成数据集上训练，直到验证损失达到指定阈值或超过最大迭代轮数。

### 随机模块拼接

`auto_train.py` 会在给定目录中自动探测可以适配的 PlugNPlay 模块，并在训练前随机抽取 `--min-modules` 到 `--max-modules` 个模块进行级联拼接。模块的随机性可以通过 `--module-seed` 控制，必要时也可以修改 `--module-root` 指定其他类别目录。

可选参数可查看 `python auto_train.py --help`。
