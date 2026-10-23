# 公开结果与来源

本目录是两轮原始实验的轻量展示副本，准确率全部来自验证集。原始 outputs 在本机保留且未修改；完整数据、权重、环境和缓存不随仓库分发。

| 路径 | 内容 | 处理方式 |
| --- | --- | --- |
| tiny_vit/metrics.csv、training_curves.png | 50 Epoch 指标与真实曲线 | 原始文件逐字节复制 |
| tiny_vit/split_indices.json | 45000 / 5000 原始索引、seed 与标签指纹 | 原样复制，ResNet18 复现直接读取此文件 |
| tiny_vit/config.json | 模型、训练与归一化配置 | 数值不变，机器专属路径改为展示标记 |
| tiny_vit/summary.json | 真实最佳 / 最低 / 最终指标与计时 | 从原摘要提取，使用相对引用 |
| resnet18/metrics.csv、training_curves.png、runtime.json | 50 Epoch 指标、曲线、运行计时与显存 | 原始文件逐字节复制 |
| resnet18/config.json、summary.json | 对照配置与真实摘要 | 同上，路径可移植化 / 摘要提取 |
| comparison/summary.json、metrics.csv、curves.png | 两轮实用比较与全部 100 行指标 | 原始文件逐字节复制 |
| comparison/inference_benchmark.json | 限定条件下合成输入推理测量 | 原始文件逐字节复制 |
| environment.json | 安装 / 运行版本、依赖检查与设备 | 本次只读核对记录 |
| manifest.json | 逐文件来源、原 / 副本 SHA256、大小和处理方式 | 本次生成的校验清单 |

每个 summary 的准确率为 0～1，README 才转换为百分比；最佳模型取最大 Validation Accuracy 的首次 Epoch。公共展示配置中的 `public_copy` 明确标识路径处理，不能冒充未修改的原始配置。原始配置的 SHA256 仍保存在 manifest 中，原始文件完整留在本机。

原始归档的项目内相对位置：

```text
outputs/task011_cifar10_vit_seed42_20261023_213930/
outputs/task012_resnet18_seed42_20261023_222908/
```

ViT 本地 best / last 各 2,904,567 bytes，ResNet18 各 89,516,167 bytes；这些权重文件均被 Git 忽略，没有复制到 results。

只读核对：

```powershell
python -B scripts/verify_portfolio.py
```

该命令使用标准库读取此目录，不要求数据集、权重或 GPU。需要在项目根目录执行；环境 / 模型 / Git 检查是显式的可选参数，详见 [复现说明](../docs/reproducibility.md)。

`.gitattributes` 对 results 下的文件关闭换行归一化，以保留来源字节与 SHA256。修改公开产物后应重新核对来源和清单，不应修改 CSV 来让指标更好看。
