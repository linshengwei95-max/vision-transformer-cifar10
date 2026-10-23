# 复现与环境说明

## 验证范围

原模型与两个训练入口保持原样。本次作品集整理只读取代码、核对 CSV / 索引 / 结果、复制轻量产物、检查依赖和发布范围，没有重训或官方测试集评估。两轮历史训练均真实完成 50 Epoch，原始命令与归档位置见 [结果说明](../results/README.md)。

只读验收入口：

```powershell
$python = Join-Path $env:USERPROFILE 'anaconda3\envs\pytorch_dev\python.exe'
& $python -X utf8 -B scripts/verify_portfolio.py --check-environment --check-models --check-ignore
```

`--check-models` 只在 CPU 创建模型、核对配置与参数量；不执行训练前向或反向。`--check-ignore` 在临时 Git 仓库进行规则检查及 `git add --dry-run`，不初始化当前项目，不提交文件。

只有保留本机原始归档时才使用 `--check-originals`；公开克隆不需要也不包含该归档。`--report` 可以把验收 JSON 写到新的本地文件，拒绝覆盖已有报告。

## 实测环境与依赖约束

| 组件 | 实测版本 |
| --- | --- |
| Python | 3.10.20 |
| PyTorch | 2.13.0+cu130 |
| torchvision | 安装元数据 0.28.0；运行时 0.28.0+cpu |
| NumPy | 2.2.6 |
| Pillow | 12.3.0 |
| Matplotlib | 3.10.9 |
| 设备 | NVIDIA GeForce RTX 5060 Laptop GPU，约 8 GB 显存 |
| 系统 | Windows，详细记录见 results/environment.json |

本机已通过依赖导入、Python 版本要求核对及 `pip check`。torchvision 的安装元数据要求 `torch==2.13.0`，与此环境匹配；其 CPU 标识和 torch 的 CUDA 标识分别记录，不混写成同一构建。实际数据预处理与 ResNet18 已用于两轮 GPU 训练。

[requirements.txt](../requirements.txt) 固定核对过的五个直接依赖公共版本；PEP 440 的公共版本约束可接受相应本地构建标识。它不锁定 CUDA / 驱动、下载源、完整传递依赖或 Conda 环境。本次还只读核对了这些具体版本的 PyPI 版本索引，没有安装、下载 wheel、创建新环境或证明任意机器都可直接安装。

已有环境不需要重装。新环境的建议顺序：

1. 准备兼容的 Python 环境；基准实验使用 3.10.20。
2. 根据 [PyTorch 官方安装说明](https://pytorch.org/get-started/locally/) 选择系统 / CPU 或 CUDA 对应的 torch 2.13.0 与 torchvision 0.28.0 二进制。
3. 在所选环境中执行下列安装与核对命令。这是提供给复现者的操作说明，本次没有执行安装。

```powershell
python -m pip install -r requirements.txt
python -m pip check
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__); print(torch.cuda.is_available())"
```

GPU 可用性应以实际检测为准。CPU 可用于产物核对和模型结构检查，不能据此复现本仓库报告的 GPU 训练耗时。新环境安装与跨机器完整训练仍未复验。

## CIFAR-10 数据准备

数据来自 [CIFAR-10 官方页面](https://www.cs.toronto.edu/~kriz/cifar.html)。由使用者获取 Python 版本并解压；仓库不分发图像或压缩包，训练入口使用 `download=False`。

`--data-root` 应指向包含下列目录的父目录：

```text
<data-root>/
└── cifar-10-batches-py/
    ├── data_batch_1 … data_batch_5
    ├── batches.meta
    └── test_batch
```

torchvision 会检查标准数据目录完整性，但项目仅构造 `CIFAR10(train=True)`，不加载官方测试集样本进行训练、调参或评估。

保存的 [split_indices.json](../results/tiny_vit/split_indices.json) 包含原始官方训练集索引，不包含图像；其 train / validation 数量为 45000 / 5000，seed=42。文件 SHA256 为 `e32757ef9a3110a34b3b240897a4ce02a27cde1f782d1ed6d6ada88722babc53`。文件原始行尾也保留，由 .gitattributes 防止 Git 转换导致校验值改变。

两轮使用相同的增强与归一化设置。模型初始化消耗的随机数不同，不能把相同 seed 和增强规则理解成逐次随机图像视图也完全相同。

## 两轮显式训练命令

项目根目录下的 PowerShell，已有环境示例：

```powershell
$python = Join-Path $env:USERPROFILE 'anaconda3\envs\pytorch_dev\python.exe'
$dataRoot = 'E:\CIFAR-10\data'
& $python -X utf8 -B train_cifar10_vit.py --help
& $python -X utf8 -B train_cifar10_resnet18.py --help
```

以下命令会启动正式训练，本次整理未执行。`--output-dir` 必须是尚不存在的目录，原入口拒绝覆盖既有实验。

```powershell
& $python -X utf8 -B train_cifar10_vit.py --train --device cuda --data-root $dataRoot --output-dir outputs/reproduce_vit_seed42 --image-size 32 --patch-size 4 --embed-dim 96 --num-heads 4 --depth 2 --mlp-ratio 4 --batch-size 64 --epochs 50 --lr 3e-4 --weight-decay 0.05 --warmup-epochs 5 --min-lr 1e-6 --seed 42 --val-size 5000 --num-workers 0

& $python -X utf8 -B train_cifar10_resnet18.py --train --device cuda --data-root $dataRoot --reference-dir results/tiny_vit --output-dir outputs/reproduce_resnet18_seed42 --batch-size 64 --epochs 50 --seed 42 --num-workers 0 --lr 0.1 --momentum 0.9 --weight-decay 5e-4 --warmup-epochs 5 --min-lr 1e-5
```

ViT 旧入口没有加载任意索引文件的命令行参数；它按 seed 生成划分，复现后应明确验证实际文件：

```powershell
& $python -X utf8 -B scripts/verify_portfolio.py --compare-split outputs/reproduce_vit_seed42/split_indices.json
```

ResNet18 旧入口默认指向本机历史输出，所以在公开复现命令中必须显式传入 `--reference-dir results/tiny_vit`。该目录含它实际需要的 config.json 和 split_indices.json；程序直接读取原索引，不重新生成。仅检查参数或产物不能被报告为重新完成训练。

## Checkpoint 与恢复

两个训练入口都保存验证准确率最优的 best 和最近的 last，包含模型、优化器、scheduler、RNG 与必要配置。恢复限定为完整 Epoch 边界，要求模型、数据指纹、划分和训练 / 调度配置一致；50 Epoch 总长度不能改成剩余 Epoch 数。

公开仓库没有权重。只有自己另行保留可信的本地 Checkpoint 才能使用原入口的 `--resume`，并显式保留原全部参数、选择新的输出目录。不要把公开展示 config 的路径替换当成可直接恢复原始二进制 Checkpoint 的保证。

## 结果解释

CSV Accuracy 范围为 0～1；README 表格转为百分比。LR 为每 Epoch 最后一个实际训练 Batch 使用的学习率，Epoch 时间为训练+验证时间，运行总时间还包括部分保存 / 记录间隔。两轮计时边界略有差异，不作为严格架构速度实验。

ResNet18 的 Epoch 49 与 50 验证准确率都是 94.88%，保存策略只在严格提高时更新，所以 best=49、last=50；最低验证 Loss 在 50。最低 Loss 与最佳 Accuracy 使用不同准则分别记录。

推理基准的硬件、Batch=64、FP32、20 次预热、100 次测量、合成输入和 CUDA Event 计时条件见 [inference_benchmark.json](../results/comparison/inference_benchmark.json)。每图数值为 Batch 摊销，不是单张请求端到端延迟，也不包含数据加载 / 预处理。
