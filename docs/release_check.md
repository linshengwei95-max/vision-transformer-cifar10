# GitHub 发布验收

## 任务 013：发布前整理范围（2026-10-23）

任务 013 完成作品集文档、轻量结果整理和发布前工程核查；不修改 Unit 01～06、两个已验收训练入口或 AGENTS.md，不删除 / 移动 / 覆盖任何原始实验文件，不重训、不评估官方测试集、不安装依赖。

项目仅包含本 ViT 学习与分类实验的整理。README 以实际代码、两轮完整 CSV 和结果归档为依据，没有新增研究成果、模型成绩或用户理解验收结论。

## 已准备的公开内容

- README：中文项目介绍与英文摘要、Mermaid 数据流、核心实现定位、配置、真实验证集对照、曲线、运行说明、限制和参考资料。
- results：指标 / 曲线 / 索引原样副本、数值配置与结果摘要、环境记录、来源 SHA256 清单。
- requirements：五个直接依赖的已核对公共版本约束；本机 pip check 通过，不把它描述为跨平台完整锁文件。
- .gitignore：排除 outputs、数据、权重、环境、缓存、临时文件和三个本地协作 / 学习记录。
- .gitattributes：结果文件保留原字节，防止换行转换破坏 SHA256；源代码 / 文档使用常规文本规则。
- scripts/verify_portfolio.py：可在没有数据和权重时核对公开结果；环境、模型与 Git 范围检查按参数启用。

## 验收方式

```powershell
$python = Join-Path $env:USERPROFILE 'anaconda3\envs\pytorch_dev\python.exe'
& $python -X utf8 -B scripts/verify_portfolio.py --check-originals --check-environment --check-models --check-ignore --compare-split outputs/task011_cifar10_vit_seed42_20261023_213930/split_indices.json
```

本机使用上述范围验收，结果包括：原始代码 / 归档 SHA256 保持一致、公开文件来源与指标一致、两组 50 Epoch / 比较 100 行完整、索引有效、文档相对链接存在、Python 语法有效、依赖一致、CPU 模型参数量匹配。

Git 文件范围在临时仓库检查：用空文件模拟被忽略的归档，复制轻量公开文件，执行 check-ignore 和 add --dry-run，并检查每个公开候选文件小于 5 MiB；临时仓库随后清理。当前项目不因此变成 Git 仓库，没有创建提交、远程或推送。实际候选清单与大小见本机 `outputs/task013_portfolio_audit/verification_final.json`；公开克隆不需要此本地审计文件。

另构造只含 32 个公开候选文件的临时副本，确认其中没有 outputs 或本地协作记录；在现有解释器下，公开产物 / 环境 / CPU 模型 / Git 规则验收通过。还用本机既有 CIFAR-10 训练数据只读验证 ResNet18 的 `--reference-dir results/tiny_vit` 所需配置与原始索引能正常加载、训练 / 验证数量为 45000 / 5000，没有初始化训练循环。该检查不是远程 clone，也不是新环境安装 / 重新训练。

训练命令仅按现有 argparse 参数核对；任务 013 没有再次运行 50 Epoch。CPU 模型检查没有执行训练前向 / 反向；任务 013 尚未进行 GitHub 网页 Mermaid 渲染验收。新机器完整安装 / 重训、不同平台数值一致性仍未验证。

## 公开发布授权与检查（2026-10-24）

作者已明确授权使用 MIT 许可证、初始化本地 Git、创建首次提交，并在当前 GitHub 登录账号下创建 Public 仓库 `vision-transformer-cifar10`、推送 `main`。目标地址为 <https://github.com/linshengwei95-max/vision-transformer-cifar10>。创建前已通过已认证 API 查询确认目标仓库不存在；若出现仓库冲突，不覆盖现有内容。

首次提交范围限定为任务 013 的 32 个公开候选文件，加上本次授权新增的 MIT `LICENSE`，合计 33 个文件。仅调整 README 与本说明的许可证 / 发布表述，不修改模型、训练脚本、指标或曲线。

发布前重新核对公开产物与原始 SHA256、100 行训练指标、45000 / 5000 索引及本机依赖，检查敏感信息和图片元数据。未发现密码、令牌、私钥、私人邮箱或个人目录路径；通用数据目录示例不含用户身份。数值指标及索引中的数字匹配已排除为扫描误报，未改写真实结果。提交作者使用 GitHub 账号和 GitHub 提供的 noreply 邮箱。

`.gitignore` 排除原始 outputs、模型权重、数据集、环境、缓存和三个本地管理文件；暂存采用明确文件清单并复核 Git 索引。只有远程创建与推送成功后，才能报告发布完成。网页验收应实际打开仓库，检查 README、Mermaid 架构图、三张实验曲线，以及文档 / 代码 / 结果链接；网页状态以远程仓库实际显示为准。

许可证采用 GitHub API 提供的 MIT 标准文本，版权署名使用公开 GitHub 账号。没有新增训练、官方测试集评估或依赖安装；后续实验仍需另行授权。
