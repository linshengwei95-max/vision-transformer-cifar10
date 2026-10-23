"""任务 012：直接复用任务 011 索引，训练 CIFAR-10 ResNet18；不加载测试集。

复用已有逐 Epoch 训练/验证、指标、Checkpoint 与 RNG 恢复函数，不修改 ViT 入口。
默认只做新 ResNet 的真实 Batch 检查，显式 --train 才执行完整训练。
"""

import argparse
import copy
import csv
from datetime import datetime
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import platform
import shutil
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, models, transforms

from train_cifar10_vit import (
    CSV_FIELDS, TransformedSubset, lr_factor, read_checkpoint, restore_rng,
    rng_state, seed_everything, seed_worker, synchronize, train as train_epochs,
)


PROJECT_DIR = Path(__file__).resolve().parent
REFERENCE_DIR = PROJECT_DIR / "outputs" / "task011_cifar10_vit_seed42_20261023_213930"


def build_data(args, device):
    reference = args.reference_dir.resolve()
    split_path = reference / "split_indices.json"
    # 缺文件直接报错；不调用 make_split，也不根据相同 seed 重建。
    split_bytes = split_path.read_bytes()
    split = json.loads(split_bytes)
    reference_config = json.loads((reference / "config.json").read_text(encoding="utf-8"))
    split_hash = hashlib.sha256(json.dumps(split, sort_keys=True).encode()).hexdigest()
    if split_hash != reference_config["split_sha256"]:
        raise ValueError("任务 011 的索引文件与其配置指纹不一致")
    if (split["dataset"] != "CIFAR10" or split["official_split"] != "train"
            or split["dataset_size"] != 50000 or split["seed"] != args.seed):
        raise ValueError("索引来源、数据规模或 seed 不符合任务要求")
    train_indices, val_indices = split["train_indices"], split["validation_indices"]
    for indices, size in ((train_indices, 45000), (val_indices, 5000)):
        if len(indices) != size or len(set(indices)) != size:
            raise ValueError("索引数量不正确或含重复项")
        if any(type(index) is not int or not 0 <= index < 50000 for index in indices):
            raise ValueError("索引必须是官方训练集范围内的整数")
    if set(train_indices).intersection(val_indices) or set(train_indices).union(val_indices) != set(range(50000)):
        raise ValueError("训练 / 验证索引重叠或没有完整覆盖官方训练集")

    source = datasets.CIFAR10(root=str(args.data_root.resolve()), train=True, download=False)
    if len(source) != 50000 or hashlib.sha256(bytes(source.targets)).hexdigest() != split["targets_sha256"]:
        raise ValueError("当前 CIFAR-10 与任务 011 的数据指纹不符")
    normalization = reference_config["normalization"]
    common = [transforms.ToTensor(), transforms.Normalize(tuple(normalization["mean"]), tuple(normalization["std"]))]
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), *common,
    ])
    val_transform = transforms.Compose(common)
    if (str(train_transform) != reference_config["train_transform"]
            or str(val_transform) != reference_config["validation_transform"]):
        raise ValueError("预处理配置与任务 011 不一致")
    if (args.batch_size != reference_config["training"]["batch_size"]
            or args.epochs != reference_config["training"]["epochs"]
            or args.seed != reference_config["training"]["seed"]):
        raise ValueError("本次对照要求 Batch Size、Epoch 数与 seed 与任务 011 一致")
    train_generator = torch.Generator().manual_seed(args.seed + 1)
    val_generator = torch.Generator().manual_seed(args.seed + 2)
    options = dict(batch_size=args.batch_size, num_workers=args.num_workers,
                   pin_memory=device.type == "cuda", worker_init_fn=seed_worker)
    return {
        "train_loader": DataLoader(TransformedSubset(source, train_indices, train_transform),
                                   shuffle=True, generator=train_generator, **options),
        "val_loader": DataLoader(TransformedSubset(source, val_indices, val_transform),
                                 shuffle=False, generator=val_generator, **options),
        "train_generator": train_generator, "val_generator": val_generator,
        "split": split, "split_sha256": split_hash, "split_bytes": split_bytes,
        "split_source": str(split_path), "split_file_sha256": hashlib.sha256(split_bytes).hexdigest(),
        "normalization": normalization, "train_transform": str(train_transform),
        "validation_transform": str(val_transform), "reference_config": reference_config,
    }


def build_model():
    # 保留 torchvision 的四组 BasicBlock；只调整 CIFAR-10 的 stem 和分类头。
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model


def build_optimizer_scheduler(model, args, batches):
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay, nesterov=False)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: lr_factor(step, args.epochs * batches,
                                         args.warmup_epochs * batches, args.min_lr / args.lr))
    return optimizer, scheduler


def preflight(model, args, device, data):
    """真实 Batch 在可丢弃副本上运行；还原 RNG，不改变正式模型或 BN 状态。"""
    original_rng = rng_state(data)
    probe = copy.deepcopy(model).train()
    optimizer, scheduler = build_optimizer_scheduler(probe, args, len(data["train_loader"]))
    try:
        images, labels = next(iter(data["train_loader"]))
        assert images.shape == (args.batch_size, 3, 32, 32) and labels.shape == (args.batch_size,)
        assert images.dtype == torch.float32 and labels.dtype == torch.int64
        assert torch.isfinite(images).all() and 0 <= labels.min() <= labels.max() <= 9
        mean = torch.tensor(data["normalization"]["mean"]).view(1, 3, 1, 1)
        std = torch.tensor(data["normalization"]["std"]).view(1, 3, 1, 1)
        pixels = images * std + mean
        assert pixels.min() >= -1e-6 and pixels.max() <= 1 + 1e-6
        before = probe.conv1.weight.detach().clone()
        logits = probe(images.to(device))
        assert logits.shape == (args.batch_size, 10) and torch.isfinite(logits).all()
        loss = nn.functional.cross_entropy(logits, labels.to(device))
        loss.backward()
        assert torch.isfinite(loss)
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in probe.parameters())
        optimizer.step()
        scheduler.step()
        assert not torch.equal(before, probe.conv1.weight)
        assert all("momentum_buffer" in state and torch.isfinite(state["momentum_buffer"]).all()
                   for state in optimizer.state.values())
        assert all(torch.isfinite(value).all() for value in probe.state_dict().values())
        probe.eval()
        with torch.no_grad():
            assert torch.isfinite(probe(images[:2].to(device))).all()
        # deepcopy 的所有参数与 buffer 独立；主模型没有参与前向/反向。
        assert all(a.data_ptr() != b.data_ptr() for a, b in zip(model.parameters(), probe.parameters()))
        assert all(parameter.grad is None for parameter in model.parameters())
        assert model.bn1.num_batches_tracked.item() == 0
        report = {"passed": True, "images_shape": list(images.shape), "labels_shape": list(labels.shape),
                  "logits_shape": list(logits.shape), "label_range": [labels.min().item(), labels.max().item()],
                  "loss": loss.item(), "finite_gradient_tensors": len(list(probe.parameters())),
                  "sgd_momentum_state_count": len(optimizer.state), "probe_parameters_updated": True,
                  "main_model_and_bn_unchanged": True, "rng_restored": True}
    finally:
        restore_rng(original_rng, data)
        del probe, optimizer, scheduler
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return report


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", action="store_true", help="明确启动 50 Epoch 正式训练")
    parser.add_argument("--data-root", type=Path, default=Path("E:/CIFAR-10/data"))
    parser.add_argument("--reference-dir", type=Path, default=REFERENCE_DIR)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", type=Path, help="本入口生成、用户可信的 Epoch 边界 Checkpoint")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    for flag, default in (("batch-size", 64), ("epochs", 50), ("warmup-epochs", 5), ("seed", 42), ("num-workers", 0)):
        parser.add_argument("--" + flag, type=int, default=default)
    for flag, default in (("lr", 0.1), ("momentum", 0.9), ("weight-decay", 5e-4), ("min-lr", 1e-5)):
        parser.add_argument("--" + flag, type=float, default=default)
    args = parser.parse_args()
    if args.resume and not args.train:
        parser.error("--resume 必须与 --train 一起使用")
    if args.batch_size != 64 or args.epochs != 50 or args.seed != 42 or args.num_workers < 0:
        parser.error("本次对照固定 Batch=64、Epochs=50、seed=42，num_workers 不能为负")
    if not 0 <= args.warmup_epochs < args.epochs:
        parser.error("要求 0 <= warmup_epochs < epochs")
    if (not all(math.isfinite(value) for value in (args.lr, args.momentum, args.weight_decay, args.min_lr))
            or args.lr <= 0 or not 0 <= args.momentum < 1 or args.weight_decay < 0
            or not 0 <= args.min_lr <= args.lr):
        parser.error("优化器 / 学习率参数不合法")
    return args


def main():
    args = parse_args()
    seed_everything(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA 实际不可用；不擅自切换设备或环境")
    device = torch.device(args.device)
    data = build_data(args, device)
    model = build_model().to(device)
    assert [len(getattr(model, f"layer{index}")) for index in range(1, 5)] == [2, 2, 2, 2]
    assert model.conv1.kernel_size == (3, 3) and model.conv1.stride == (1, 1)
    assert isinstance(model.maxpool, nn.Identity) and model.fc.out_features == 10
    optimizer, scheduler = build_optimizer_scheduler(model, args, len(data["train_loader"]))
    run_dir = (args.output_dir or PROJECT_DIR / "outputs" / (
        "task012_resnet18_seed42_" + datetime.now().strftime("%Y%m%d_%H%M%S"))).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "checkpoints").mkdir()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(run_dir / "run.log", encoding="utf-8"), logging.StreamHandler()])
    environment = {"python_executable": sys.executable, "python": platform.python_version(),
                   "torch": torch.__version__, "torchvision": torchvision.__version__, "device": str(device),
                   "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                   "gpu_total_bytes": torch.cuda.get_device_properties(device).total_memory if device.type == "cuda" else None,
                   "disk_free_bytes": shutil.disk_usage(run_dir).free}
    config = {"mode": "train" if args.train else "preflight-only",
              "model": {"architecture": "torchvision.resnet18", "weights": None, "image_size": 32,
                        "conv1": {"kernel_size": 3, "stride": 1, "padding": 1, "bias": False},
                        "maxpool": "Identity", "num_classes": 10, "blocks": [2, 2, 2, 2]},
              "training": {key: getattr(args, key) for key in (
                  "batch_size", "epochs", "warmup_epochs", "seed", "num_workers", "lr", "momentum", "weight_decay", "min_lr")},
              "optimizer": "SGD", "nesterov": False, "loss": "CrossEntropyLoss",
              "scheduler": "per-step warmup+cosine", "data_root": str(args.data_root.resolve()),
              "real_data": True, "split_sha256": data["split_sha256"], "split_file_sha256": data["split_file_sha256"],
              "split_source": data["split_source"], "split_method": "read saved task011 indices; never regenerated",
              "reference_dir": str(args.reference_dir.resolve()), "normalization": data["normalization"],
              "train_transform": data["train_transform"], "validation_transform": data["validation_transform"],
              "num_parameters": sum(p.numel() for p in model.parameters()), "accuracy_scale": "0..1",
              "csv_learning_rate": "last optimizer batch in epoch", "environment": environment,
              "resume_from": str(args.resume.resolve()) if args.resume else None,
              "comparison_scope": "same indices/preprocessing/epochs/batch/seed; different model capacity and optimizer recipe; not architecture-only ablation"}
    (run_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (run_dir / "split_indices.json").write_bytes(data["split_bytes"])
    assert (run_dir / "split_indices.json").read_bytes() == Path(data["split_source"]).read_bytes()
    with (run_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        csv.DictWriter(stream, fieldnames=CSV_FIELDS).writeheader()
    hashes = [{"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
              for path in [PROJECT_DIR / "AGENTS.md", *sorted(PROJECT_DIR.glob("*.py")),
                           *sorted(args.reference_dir.rglob("*"))] if path.is_file()]
    (run_dir / "source_hashes.json").write_text(json.dumps(hashes, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    command = "& '" + sys.executable + "' -X utf8 -B '" + str(Path(__file__).resolve()) + "' "
    command += " ".join("'" + argument.replace("'", "''") + "'" for argument in sys.argv[1:])
    (run_dir / "command.ps1").write_text(command + "\n", encoding="utf-8")
    logging.info("Mode=%s device=%s GPU=%s parameters=%s output=%s", config["mode"], device,
                 environment["gpu_name"], config["num_parameters"], run_dir)
    logging.info("Saved task011 indices reused: 45000/5000; file SHA256=%s", data["split_file_sha256"])
    report = preflight(model, args, device, data)
    (run_dir / "preflight_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    logging.info("ResNet real-batch preflight PASS: %s -> %s loss=%.6f; SGD state=%s; main model/RNG unchanged",
                 report["images_shape"], report["logits_shape"], report["loss"], report["sgd_momentum_state_count"])
    if not args.train:
        return
    resume = read_checkpoint(args.resume) if args.resume else None
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = datetime.now().isoformat()
    start = time.perf_counter()
    status = "failed"
    try:
        logging.info("Formal 50-epoch training START; optimizer=SGD momentum=%.2f lr=%.6g wd=%.6g min_lr=%.6g",
                     args.momentum, args.lr, args.weight_decay, args.min_lr)
        train_epochs(model, optimizer, scheduler, data, args, device, config, run_dir, resume)
        status = "completed"
    finally:
        synchronize(device)
        runtime = {"status": status, "started_at": started, "ended_at": datetime.now().isoformat(),
                   "train_loop_wall_seconds": time.perf_counter() - start,
                   "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                   "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
                   "peak_scope": "formal train loop only; excludes disposable preflight; not whole-card driver usage",
                   "official_test_evaluated": False}
        (run_dir / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n", encoding="utf-8")
    logging.info("Formal training COMPLETE in %.3f seconds; peak allocated=%s bytes",
                 runtime["train_loop_wall_seconds"], runtime["cuda_peak_allocated_bytes"])


if __name__ == "__main__":
    main()
