"""CIFAR-10 / 小型 ViT 训练管线。默认仅冒烟检查，显式 --train 才训练。

只读取官方训练集，用固定索引划分训练/验证；从不下载数据或读取测试集样本。
恢复点位于完整 Epoch 边界，不支持中途 Batch 恢复。
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
import random
import shutil
import sys
import tempfile
import time

# 必须在首次 CUDA 运算之前设置，配合确定性算法固定随机实验。
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import datasets, transforms

from unit06_vision_transformer import VisionTransformer


PROJECT_DIR = Path(__file__).resolve().parent
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)
MODEL_KEYS = ("image_size", "patch_size", "in_channels", "embed_dim", "num_heads",
              "depth", "mlp_ratio", "num_classes")
TRAIN_KEYS = ("lr", "weight_decay", "batch_size", "epochs", "warmup_epochs",
              "min_lr", "seed", "val_size", "num_workers")
CSV_FIELDS = ("epoch", "train_loss", "train_accuracy", "validation_loss",
              "validation_accuracy", "learning_rate", "epoch_seconds")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def seed_worker(worker_id):
    # Windows spawn 下使用顶层函数；worker 的种子由 DataLoader generator 派生。
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


class TransformedSubset(Dataset):
    """共用只读原图，但训练和验证使用各自的索引与 transform。"""

    def __init__(self, source, indices, transform):
        self.source, self.indices, self.transform = source, indices, transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        image, label = self.source[self.indices[index]]
        return self.transform(image), label


def make_split(size, val_size, seed):
    if not 0 < val_size < size:
        raise ValueError("验证集大小必须介于 0 和数据集大小之间")
    order = torch.randperm(size, generator=torch.Generator().manual_seed(seed)).tolist()
    train_indices, val_indices = order[val_size:], order[:val_size]
    assert len(set(train_indices)) == len(train_indices)
    assert len(set(val_indices)) == len(val_indices)
    assert not set(train_indices).intersection(val_indices)
    assert set(train_indices).union(val_indices) == set(range(size))
    return train_indices, val_indices


def build_data(args, device):
    candidates = ([args.data_root] if args.data_root else
                  [PROJECT_DIR / "data", Path("E:/CIFAR-10/data")])
    attempts, source, data_root = [], None, None
    for candidate in candidates:
        candidate = Path(candidate).resolve()
        try:
            source = datasets.CIFAR10(root=str(candidate), train=True, download=False)
            data_root = candidate
            break
        except (RuntimeError, FileNotFoundError) as error:
            attempts.append({"path": str(candidate), "error": str(error)})
    real_data = source is not None
    if not real_data:
        if args.train:
            raise RuntimeError(f"没有可读取的 CIFAR-10；禁止自动下载或用模拟数据正式训练：{attempts}")
        # 缺数据时仅检查接口，不冒充 CIFAR-10 数据或真实实验结果。
        source = datasets.FakeData(size=50000, image_size=(3, 32, 32), num_classes=10,
                                   random_offset=args.seed)
    if len(source) != 50000:
        raise ValueError("预期 CIFAR-10 官方训练集包含 50000 张图像")

    train_indices, val_indices = make_split(len(source), args.val_size, args.seed)
    resized = [transforms.Resize((args.image_size, args.image_size))] if args.image_size != 32 else []
    common = [*resized, transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), *common,
    ])
    val_transform = transforms.Compose(common)
    train_dataset = TransformedSubset(source, train_indices, train_transform)
    val_dataset = TransformedSubset(source, val_indices, val_transform)
    train_generator = torch.Generator().manual_seed(args.seed + 1)
    val_generator = torch.Generator().manual_seed(args.seed + 2)
    loader_options = dict(batch_size=args.batch_size, num_workers=args.num_workers,
                          pin_memory=device.type == "cuda", worker_init_fn=seed_worker)
    train_loader = DataLoader(train_dataset, shuffle=True, generator=train_generator, **loader_options)
    val_loader = DataLoader(val_dataset, shuffle=False, generator=val_generator, **loader_options)
    split = {"dataset": "CIFAR10" if real_data else "FakeData-interface-only",
             "official_split": "train" if real_data else None,
             "seed": args.seed, "dataset_size": len(source),
             "targets_sha256": hashlib.sha256(bytes(source.targets)).hexdigest() if real_data else None,
             "train_indices": train_indices, "validation_indices": val_indices}
    fingerprint = hashlib.sha256(json.dumps(split, sort_keys=True).encode()).hexdigest()
    return {"train_loader": train_loader, "val_loader": val_loader,
            "train_generator": train_generator, "val_generator": val_generator,
            "split": split, "split_sha256": fingerprint, "real_data": real_data,
            "data_root": str(data_root) if data_root else None, "attempts": attempts,
            "train_transform": str(train_transform), "validation_transform": str(val_transform)}


def lr_factor(step, total_steps, warmup_steps, min_ratio):
    if warmup_steps and step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = min(1.0, max(0.0, (step - warmup_steps) / max(1, total_steps - warmup_steps - 1)))
    return min_ratio + (1 - min_ratio) * (1 + math.cos(math.pi * progress)) / 2


def build_optimizer_scheduler(model, args, batches_per_epoch):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: lr_factor(step, args.epochs * batches_per_epoch,
                                         args.warmup_epochs * batches_per_epoch,
                                         args.min_lr / args.lr))
    return optimizer, scheduler


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_pass(model, loader, device, optimizer=None, scheduler=None,
             max_batches=None, update_parameters=True):
    """指标按样本数加权；accuracy 为 0～1；LR 记录最后一个实际 Batch 的值。"""
    training, previous_mode = optimizer is not None, model.training
    model.train(training)
    total_loss, total_correct, total_samples = 0.0, 0, 0
    learning_rate = optimizer.param_groups[0]["lr"] if training else None
    synchronize(device)
    start = time.perf_counter()
    try:
        with torch.set_grad_enabled(training):
            for batch_index, (images, labels) in enumerate(loader):
                images, labels = images.to(device), labels.to(device)
                if training:
                    optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                loss = nn.functional.cross_entropy(logits, labels)
                if not torch.isfinite(logits).all() or not torch.isfinite(loss):
                    raise FloatingPointError("Logits 或 Loss 出现 NaN/Inf")
                if training:
                    loss.backward()
                    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
                        raise FloatingPointError("存在缺失或非有限的梯度")
                    learning_rate = optimizer.param_groups[0]["lr"]
                    if update_parameters:
                        optimizer.step()
                        scheduler.step()
                count = labels.size(0)
                total_loss += loss.item() * count
                total_correct += (logits.argmax(dim=-1) == labels).sum().item()
                total_samples += count
                if max_batches is not None and batch_index + 1 >= max_batches:
                    break
    finally:
        model.train(previous_mode)
    synchronize(device)
    if total_samples == 0:
        raise ValueError("DataLoader 为空")
    return {"loss": total_loss / total_samples, "accuracy": total_correct / total_samples,
            "samples": total_samples, "learning_rate": learning_rate,
            "seconds": time.perf_counter() - start}


def rng_state(data):
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "train_generator": data["train_generator"].get_state(),
            "val_generator": data["val_generator"].get_state()}


def restore_rng(state, data):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])
    data["train_generator"].set_state(state["train_generator"].cpu())
    data["val_generator"].set_state(state["val_generator"].cpu())


def checkpoint_state(model, optimizer, scheduler, epoch, best_accuracy, config, data):
    return {"schema_version": 1, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "epoch": epoch, "best_validation_accuracy": best_accuracy,
            "config": config, "rng": rng_state(data)}


def save_checkpoint(path, state):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(state, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def read_checkpoint(path):
    # 文件含 Python/NumPy RNG 状态；只对本脚本创建、用户可信的本地文件调用。
    return torch.load(path, map_location="cpu", weights_only=False)


def restore_checkpoint(state, model, optimizer, scheduler, data):
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    restore_rng(state["rng"], data)
    return state["epoch"] + 1, state["best_validation_accuracy"]


def assert_nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left.cpu(), right.cpu(), rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_nested_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for item_left, item_right in zip(left, right):
            assert_nested_equal(item_left, item_right)
    else:
        assert left == right


def check_checkpoint(model, args, optimizer, config, data, run_dir):
    """仅在丢弃的模型副本上探测 AdamW 更新/恢复；主模型与 RNG 保持原状。"""
    original_rng = rng_state(data)
    probe = copy.deepcopy(model)
    probe_optimizer, probe_scheduler = build_optimizer_scheduler(probe, args, len(data["train_loader"]))
    for original, parameter in zip(model.parameters(), probe.parameters()):
        parameter.grad = original.grad.detach().clone()
    probe_optimizer.step()  # 用已有梯度建立真实的 AdamW 一、二阶矩状态。
    probe_scheduler.step()
    assert probe_optimizer.state
    checkpoint_bytes = 0
    temporary_path = None
    try:
        with tempfile.TemporaryDirectory(prefix="checkpoint_probe_", dir=run_dir) as temporary:
            temporary_path = Path(temporary).resolve()
            assert temporary_path.is_relative_to(run_dir.resolve())
            checkpoint_path = temporary_path / "probe.pt"
            save_checkpoint(checkpoint_path, checkpoint_state(
                probe, probe_optimizer, probe_scheduler, -1, None, config, data))
            checkpoint_bytes = checkpoint_path.stat().st_size
            expected_rng = (random.random(), float(np.random.rand()), torch.rand(3),
                            torch.rand(3, device=next(model.parameters()).device),
                            torch.rand(3, generator=data["train_generator"]),
                            torch.rand(3, generator=data["val_generator"]))
            restored = copy.deepcopy(model)
            restored_optimizer, restored_scheduler = build_optimizer_scheduler(
                restored, args, len(data["train_loader"]))
            next_epoch, best_accuracy = restore_checkpoint(
                read_checkpoint(checkpoint_path), restored, restored_optimizer, restored_scheduler, data)
            assert next_epoch == 0 and best_accuracy is None  # 没有伪造已完成的 Epoch。
            actual_rng = (random.random(), float(np.random.rand()), torch.rand(3),
                          torch.rand(3, device=next(model.parameters()).device),
                          torch.rand(3, generator=data["train_generator"]),
                          torch.rand(3, generator=data["val_generator"]))
            assert_nested_equal(expected_rng, actual_rng)
            assert_nested_equal(probe.state_dict(), restored.state_dict())
            assert_nested_equal(probe_optimizer.state_dict(), restored_optimizer.state_dict())
            assert_nested_equal(probe_scheduler.state_dict(), restored_scheduler.state_dict())
            # 给两份副本相同梯度，恢复后的下一次优化结果也必须完全一致。
            for original, parameter in zip(model.parameters(), restored.parameters()):
                parameter.grad = original.grad.detach().clone()
            for current_optimizer, current_scheduler in (
                (probe_optimizer, probe_scheduler), (restored_optimizer, restored_scheduler)
            ):
                current_optimizer.step()
                current_scheduler.step()
            assert_nested_equal(probe.state_dict(), restored.state_dict())
            assert_nested_equal(probe_optimizer.state_dict(), restored_optimizer.state_dict())
            assert_nested_equal(probe_scheduler.state_dict(), restored_scheduler.state_dict())
    finally:
        restore_rng(original_rng, data)
    assert temporary_path is not None and not temporary_path.exists()
    assert_nested_equal(optimizer.state_dict()["state"], {})  # 主优化器没有更新。
    return {"passed": True, "bytes": checkpoint_bytes, "temporary_files_cleaned": True,
            "nonempty_adamw_state": True, "next_update_identical": True, "rng_restored": True}


def check_batch(loader, args):
    images, labels = next(iter(loader))
    assert tuple(images.shape) == (args.batch_size, 3, args.image_size, args.image_size)
    assert tuple(labels.shape) == (args.batch_size,)
    assert images.dtype == torch.float32 and labels.dtype == torch.int64
    assert torch.isfinite(images).all() and 0 <= labels.min() <= labels.max() <= 9
    mean, std = torch.tensor(MEAN).view(1, 3, 1, 1), torch.tensor(STD).view(1, 3, 1, 1)
    pixels = images * std + mean
    assert pixels.min() >= -1e-6 and pixels.max() <= 1 + 1e-6
    return {"images_shape": list(images.shape), "labels_shape": list(labels.shape),
            "images_dtype": str(images.dtype), "labels_dtype": str(labels.dtype),
            "normalized_range": [images.min().item(), images.max().item()],
            "pixel_range": [pixels.min().item(), pixels.max().item()],
            "label_range": [labels.min().item(), labels.max().item()]}


def smoke_test(model, optimizer, scheduler, data, args, device, config, run_dir):
    initial = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    train_batch = check_batch(data["train_loader"], args)
    val_batch = check_batch(data["val_loader"], args)
    # 验证 transform 没有随机增强，并核对保存的划分可重建。
    val_dataset = data["val_loader"].dataset
    assert torch.equal(val_dataset[0][0], val_dataset[0][0])
    train_indices, val_indices = make_split(50000, args.val_size, args.seed)
    assert train_indices == data["split"]["train_indices"]
    assert val_indices == data["split"]["validation_indices"]
    total_steps, warmup_steps = args.epochs * len(data["train_loader"]), args.warmup_epochs * len(data["train_loader"])
    if warmup_steps:
        assert math.isclose(lr_factor(0, total_steps, warmup_steps, args.min_lr / args.lr), 1 / warmup_steps)
        assert lr_factor(warmup_steps - 1, total_steps, warmup_steps, args.min_lr / args.lr) == 1
    if total_steps > 1:
        assert math.isclose(lr_factor(total_steps - 1, total_steps, warmup_steps, args.min_lr / args.lr), args.min_lr / args.lr)
    modes = []
    hook = model.register_forward_pre_hook(lambda module, inputs: modes.append((module.training, torch.is_grad_enabled())))
    model.eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    try:
        train_result = run_pass(model, data["train_loader"], device, optimizer, scheduler,
                                max_batches=1, update_parameters=False)
        assert modes[-1] == (True, True) and not model.training
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        gradients = {name: parameter.grad.clone() for name, parameter in model.named_parameters()}
        train_peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        model.train()
        val_result = run_pass(model, data["val_loader"], device, max_batches=1)
        assert modes[-1] == (False, False) and model.training
        for name, parameter in model.named_parameters():
            assert torch.equal(parameter.grad, gradients[name])
        checkpoint_report = check_checkpoint(model, args, optimizer, config, data, run_dir)
        for name, parameter in model.named_parameters():
            assert torch.equal(parameter.detach(), initial[name])
    finally:
        hook.remove()
    return {"mode": "smoke-only", "real_data": data["real_data"], "completed_epochs": 0,
            "official_test_evaluated": False, "main_model_updated": False,
            "train_batch": train_batch, "validation_batch": val_batch,
            "train_forward_loss_backward": train_result, "validation_one_batch_only": val_result,
            "finite_gradient_tensors": len(list(model.parameters())),
            "train_eval_modes_correct": True, "validation_did_not_change_gradients": True,
            "split": {"train": len(train_indices), "validation": len(val_indices),
                      "overlap": 0, "full_coverage": 50000, "reproducible": True},
            "scheduler_boundaries_passed": True, "checkpoint": checkpoint_report,
            "train_batch_peak_allocated_bytes": train_peak, "checks_passed": True}


def train(model, optimizer, scheduler, data, args, device, config, run_dir, resume_state):
    start_epoch, best_accuracy = 0, None
    if resume_state is not None:
        if resume_state["config"]["model"] != config["model"] or resume_state["config"]["training"] != config["training"]:
            raise ValueError("恢复训练的模型、种子、优化器、Epoch 总数和调度配置必须与 Checkpoint 一致")
        if resume_state["config"]["split_sha256"] != data["split_sha256"]:
            raise ValueError("恢复训练的训练/验证索引或数据指纹不一致")
        start_epoch, best_accuracy = restore_checkpoint(resume_state, model, optimizer, scheduler, data)
        # 新实验目录保留上次的 best（若它与恢复点的 best 分数一致）。不改写源实验。
        previous_best = args.resume.resolve().parent / "best.pt"
        if previous_best.exists():
            best_state = read_checkpoint(previous_best)
            if (best_state["epoch"] <= resume_state["epoch"]
                    and best_state["best_validation_accuracy"] == best_accuracy
                    and best_state["config"]["split_sha256"] == data["split_sha256"]):
                shutil.copy2(previous_best, run_dir / "checkpoints" / "best.pt")
        logging.info("Resume at epoch %s; historical best validation accuracy=%s", start_epoch + 1, best_accuracy)
    if start_epoch >= args.epochs:
        raise ValueError("Checkpoint 已达到配置的总 Epoch 数，没有剩余训练")
    with (run_dir / "metrics.csv").open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        for epoch in range(start_epoch, args.epochs):
            synchronize(device)
            start = time.perf_counter()
            train_result = run_pass(model, data["train_loader"], device, optimizer, scheduler)
            val_result = run_pass(model, data["val_loader"], device)
            synchronize(device)
            row = {"epoch": epoch + 1, "train_loss": train_result["loss"],
                   "train_accuracy": train_result["accuracy"], "validation_loss": val_result["loss"],
                   "validation_accuracy": val_result["accuracy"], "learning_rate": train_result["learning_rate"],
                   "epoch_seconds": time.perf_counter() - start}
            improved = best_accuracy is None or val_result["accuracy"] > best_accuracy
            if improved:
                best_accuracy = val_result["accuracy"]
            state = checkpoint_state(model, optimizer, scheduler, epoch, best_accuracy, config, data)
            save_checkpoint(run_dir / "checkpoints" / "last.pt", state)
            if improved:
                save_checkpoint(run_dir / "checkpoints" / "best.pt", state)
            writer.writerow(row)
            stream.flush()
            logging.info("Epoch %s/%s train_loss=%.6f train_acc=%.4f val_loss=%.6f val_acc=%.4f lr=%.8g seconds=%.2f",
                         epoch + 1, args.epochs, row["train_loss"], row["train_accuracy"], row["validation_loss"],
                         row["validation_accuracy"], row["learning_rate"], row["epoch_seconds"])


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--train", action="store_true", help="显式允许完整 Epoch 训练")
    mode.add_argument("--smoke-test", action="store_true", help="默认行为：仅批次冒烟检查")
    parser.add_argument("--data-root", type=Path, help="包含 cifar-10-batches-py 的目录；绝不自动下载")
    parser.add_argument("--output-dir", type=Path, help="新建输出目录，拒绝覆盖已有目录")
    parser.add_argument("--resume", type=Path, help="本脚本生成的可信 Checkpoint；只与 --train 合用")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    for flag, default in (("image-size", 32), ("patch-size", 4), ("embed-dim", 96),
                          ("num-heads", 4), ("depth", 2), ("batch-size", 64),
                          ("epochs", 50), ("warmup-epochs", 5), ("seed", 42),
                          ("val-size", 5000), ("num-workers", 0)):
        parser.add_argument("--" + flag, type=int, default=default)
    parser.add_argument("--mlp-ratio", type=float, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    args = parser.parse_args()
    args.in_channels, args.num_classes = 3, 10
    if args.resume and not args.train:
        parser.error("--resume 必须与显式 --train 一起使用")
    if args.batch_size <= 0 or args.batch_size > min(50000 - args.val_size, args.val_size):
        parser.error("Batch Size 必须为正数且不大于训练/验证集大小")
    if args.epochs <= 0 or not 0 <= args.warmup_epochs < args.epochs:
        parser.error("要求 epochs > 0 且 0 <= warmup_epochs < epochs")
    if args.num_workers < 0 or not 0 <= args.seed < 2**32 - 2:
        parser.error("num_workers 不能为负；seed 必须处于 [0, 2**32-3]")
    if not all(math.isfinite(value) for value in (args.lr, args.weight_decay, args.min_lr)):
        parser.error("学习率与 Weight Decay 必须有限")
    if args.lr <= 0 or args.weight_decay < 0 or not 0 <= args.min_lr <= args.lr:
        parser.error("要求 lr > 0、weight_decay >= 0、0 <= min_lr <= lr")
    return args


def main():
    args = parse_args()
    seed_everything(args.seed)
    available = torch.cuda.is_available()
    if args.device == "cuda" and not available:
        raise RuntimeError("指定了 CUDA，但 torch.cuda.is_available() 为 False")
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and available) else "cpu")
    # 实际运行一次，CUDA 构建版本字符串不能替代设备检查。
    probe = torch.ones(1, device=device) + 1
    assert probe.item() == 2
    data = build_data(args, device)
    model_config = {key: getattr(args, key) for key in MODEL_KEYS}
    model = VisionTransformer(**model_config).to(device)
    optimizer, scheduler = build_optimizer_scheduler(model, args, len(data["train_loader"]))
    run_dir = args.output_dir or PROJECT_DIR / "outputs" / (
        "cifar10_vit_" + ("train_" if args.train else "smoke_") + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "checkpoints").mkdir()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(run_dir / "run.log", encoding="utf-8"), logging.StreamHandler()])
    environment = {"python_executable": sys.executable, "python": platform.python_version(),
                   "torch": torch.__version__, "torchvision": torchvision.__version__,
                   "device": str(device), "cuda_available": available,
                   "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                   "gpu_total_bytes": torch.cuda.get_device_properties(device).total_memory if device.type == "cuda" else None,
                   "gpu_free_bytes_at_start": torch.cuda.mem_get_info(device)[0] if device.type == "cuda" else None,
                   "output_disk_free_bytes": shutil.disk_usage(run_dir).free}
    config = {"mode": "train" if args.train else "smoke-only", "model": model_config,
              "training": {key: getattr(args, key) for key in TRAIN_KEYS},
              "optimizer": "AdamW", "loss": "CrossEntropyLoss", "scheduler": "per-step warmup+cosine",
              "data_root": data["data_root"], "real_data": data["real_data"],
              "dataset_attempts": data["attempts"], "split_sha256": data["split_sha256"],
              "normalization": {"mean": MEAN, "std": STD}, "accuracy_scale": "0..1",
              "csv_learning_rate": "last optimizer batch in epoch", "num_parameters": sum(p.numel() for p in model.parameters()),
              "train_transform": data["train_transform"], "validation_transform": data["validation_transform"],
              "environment": environment, "resume_from": str(args.resume.resolve()) if args.resume else None,
              "resume_scope": "complete epoch boundary; same configuration and split; same software/device for exact replay"}
    (run_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (run_dir / "split_indices.json").write_text(json.dumps(data["split"], indent=2) + "\n", encoding="utf-8")
    with (run_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        csv.DictWriter(stream, fieldnames=CSV_FIELDS).writeheader()
    logging.info("Mode=%s | device=%s | GPU=%s | CIFAR10=%s | root=%s", config["mode"], device,
                 environment["gpu_name"], data["real_data"], data["data_root"])
    logging.info("Train=%s validation=%s overlap=0 | output=%s", len(data["train_loader"].dataset),
                 len(data["val_loader"].dataset), run_dir)
    if args.train:
        resume_state = read_checkpoint(args.resume) if args.resume else None
        train(model, optimizer, scheduler, data, args, device, config, run_dir, resume_state)
    else:
        report = smoke_test(model, optimizer, scheduler, data, args, device, config, run_dir)
        report["environment"] = environment
        (run_dir / "smoke_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        logging.info("Smoke checks PASS | train F/L/B loss=%.6f samples=%s seconds=%.3f | peak allocated=%s bytes",
                     report["train_forward_loss_backward"]["loss"], report["train_forward_loss_backward"]["samples"],
                     report["train_forward_loss_backward"]["seconds"], report["train_batch_peak_allocated_bytes"])
        logging.info("Checkpoint save/load, nonempty AdamW state, next update, RNG, cleanup: PASS")
        logging.info("Completed epochs=0; main model unchanged; no official test evaluation. Report=%s", run_dir / "smoke_report.json")


if __name__ == "__main__":
    main()
