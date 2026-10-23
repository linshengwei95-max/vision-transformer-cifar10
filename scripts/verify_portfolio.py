"""核对公开结果、文档引用和发布文件范围；不训练，不读取数据集。

默认仅使用标准库。可选环境 / 模型检查需要项目依赖；Git 规则检查在临时仓库
执行，只复制轻量公开文件，用空文件模拟被忽略的本地归档，不初始化项目仓库。
"""

import argparse
import ast
import csv
import hashlib
import importlib.metadata as metadata
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        result = [{key: int(value) if key == "epoch" else float(value)
                   for key, value in row.items()} for row in csv.DictReader(stream)]
    assert [row["epoch"] for row in result] == list(range(1, 51)), path
    assert all(math.isfinite(value) for row in result for value in row.values()), path
    assert all(0 <= row[key] <= 1 for row in result for key in ("train_accuracy", "validation_accuracy")), path
    return result


def artifacts(check_originals):
    manifest = read_json(ROOT / "results/manifest.json")
    for entry in manifest["files"]:
        path = (ROOT / entry["path"]).resolve()
        assert path.is_relative_to(ROOT) and path.is_file(), entry["path"]
        assert path.stat().st_size == entry["bytes"] and digest(path) == entry["sha256"], entry["path"]
        if check_originals and entry.get("source"):
            source = (ROOT / entry["source"]).resolve()
            assert source.is_relative_to(ROOT) and source.is_file(), entry["source"]
            assert digest(source) == entry["source_sha256"], entry["source"]
            if entry["operation"] == "byte_copy":
                assert source.read_bytes() == path.read_bytes(), entry["path"]
    if check_originals:
        baseline = ROOT / "outputs/task013_portfolio_audit/original_hashes.json"
        for entry in read_json(baseline):
            assert digest(ROOT / entry["path"]) == entry["sha256"], entry["path"]

    split_path = ROOT / "results/tiny_vit/split_indices.json"
    split = read_json(split_path)
    train, val = split["train_indices"], split["validation_indices"]
    assert len(train) == len(set(train)) == 45000 and len(val) == len(set(val)) == 5000
    assert all(type(index) is int and 0 <= index < 50000 for index in train + val)
    assert set(train).isdisjoint(val) and set(train).union(val) == set(range(50000))
    assert split["seed"] == 42 and split["official_split"] == "train" and split["dataset"] == "CIFAR10"
    canonical = hashlib.sha256(json.dumps(split, sort_keys=True).encode()).hexdigest()
    configs, metrics, summaries = {}, {}, {}
    for key, name in (("tiny_vit", "Tiny ViT"), ("resnet18", "ResNet18")):
        base = ROOT / "results" / key
        config, current_rows, summary = read_json(base / "config.json"), rows(base / "metrics.csv"), read_json(base / "summary.json")
        assert config["split_sha256"] == canonical
        assert config["training"]["batch_size"] == 64 and config["training"]["epochs"] == 50 and config["training"]["seed"] == 42
        assert summary["completed_epochs"] == 50 and not summary["official_test_evaluated"]
        assert summary["best_validation"] == max(current_rows, key=lambda row: row["validation_accuracy"])
        assert summary["lowest_validation_loss"] == min(current_rows, key=lambda row: row["validation_loss"])
        assert summary["lowest_train_loss"] == min(current_rows, key=lambda row: row["train_loss"])
        assert summary["final"] == current_rows[-1]
        configs[name], metrics[name], summaries[name] = config, current_rows, summary
    for field in ("normalization", "train_transform", "validation_transform"):
        assert configs["Tiny ViT"][field] == configs["ResNet18"][field]
    assert configs["Tiny ViT"]["num_parameters"] == 235882 and configs["ResNet18"]["num_parameters"] == 11173962
    assert summaries["Tiny ViT"]["best_validation"]["validation_accuracy"] == .758
    assert summaries["ResNet18"]["best_validation"]["validation_accuracy"] == .9488
    assert summaries["ResNet18"]["best_validation"]["epoch"] == 49
    comparison = read_json(ROOT / "results/comparison/summary.json")
    assert comparison["indices_byte_identical"] and comparison["preprocessing_identical"]
    assert not comparison["official_test_evaluated"] and comparison["split_file_sha256"] == digest(split_path)
    for name in metrics:
        assert comparison["models"][name]["best_validation"] == summaries[name]["best_validation"]
        assert comparison["models"][name]["training_config"] == configs[name]["training"]
    with (ROOT / "results/comparison/metrics.csv").open(encoding="utf-8-sig", newline="") as stream:
        combined = list(csv.DictReader(stream))
    assert len(combined) == 100
    for name in metrics:
        selected = [{key: int(value) if key == "epoch" else float(value)
                     for key, value in row.items() if key != "model"} for row in combined if row["model"] == name]
        assert selected == metrics[name]
    return {"manifest_files": len(manifest["files"]), "metric_rows": 100, "train_indices": 45000,
            "validation_indices": 5000, "overlap": 0, "split_file_sha256": digest(split_path),
            "best_validation": {name: summaries[name]["best_validation"]["validation_accuracy"] for name in summaries},
            "original_archive_checked": check_originals}


def documents():
    checked = 0
    for path in [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md")), ROOT / "results/README.md"]:
        text = path.read_text(encoding="utf-8")
        assert text.count("```") % 2 == 0, path
        for target in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", text):
            target = target.strip("<>")
            if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", target) or target.startswith("#"):
                continue
            linked = (path.parent / unquote(target.split("#", 1)[0])).resolve()
            assert linked.is_relative_to(ROOT) and linked.exists(), (path, target)
            checked += 1
    for path in [*ROOT.glob("*.py"), *sorted((ROOT / "scripts").glob("*.py"))]:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {"local_links_checked": checked, "python_syntax": "passed"}


def environment(check_models):
    installed = {}
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        name, expected = line.split("==")
        version = metadata.version(name)
        assert version.split("+")[0] == expected, (name, version, expected)
        installed[name] = version
    result = subprocess.run([sys.executable, "-B", "-m", "pip", "check"], capture_output=True, text=True, check=True)
    report = {"versions": installed, "pip_check": result.stdout.strip()}
    if check_models:
        sys.path.insert(0, str(ROOT))
        from unit06_vision_transformer import VisionTransformer
        from train_cifar10_resnet18 import build_model
        config = read_json(ROOT / "results/tiny_vit/config.json")
        vit, resnet = VisionTransformer(**config["model"]), build_model()
        assert sum(p.numel() for p in vit.parameters()) == config["num_parameters"] == 235882
        assert vit.input_embedding.patch_projection.in_features == 48
        assert len(vit.blocks) == 2 and vit.blocks[0].attention.num_heads == 4
        assert sum(p.numel() for p in resnet.parameters()) == 11173962
        assert resnet.conv1.kernel_size == (3, 3) and resnet.conv1.stride == (1, 1) and resnet.fc.out_features == 10
        assert [len(getattr(resnet, f"layer{index}")) for index in range(1, 5)] == [2, 2, 2, 2]
        report["models"] = {"Tiny ViT": 235882, "ResNet18": 11173962, "scope": "CPU initialization only; no forward/backward/training"}
    return report


def git_rules():
    actual = sorted(path.relative_to(ROOT).as_posix() for path in ROOT.rglob("*")
                    if path.is_file() and ".git" not in path.relative_to(ROOT).parts)
    probes = ["AGENTS.md", "PROJECT_HANDOFF.md", "LEARNING_LOG.md",
              "data/cifar-10-batches-py/data_batch_1", ".venv/Lib/site-packages/example.py",
              "__pycache__/example.pyc", "results/accidental_weight.pt", "weights/model.pth",
              "temp.tmp", "dataset_archive.zip", "outputs/future_run/checkpoints/best.pt"]
    paths = sorted(set(actual + probes))
    with tempfile.TemporaryDirectory(prefix="vit_portfolio_") as temporary:
        scratch = Path(temporary).resolve()
        assert scratch.is_relative_to(Path(tempfile.gettempdir()).resolve()) and scratch.name.startswith("vit_portfolio_")
        subprocess.run(["git", "-C", str(scratch), "init", "--quiet"], check=True, capture_output=True)
        for relative in paths:
            path = scratch / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        for name in (".gitignore", ".gitattributes"):
            shutil.copyfile(ROOT / name, scratch / name)
        ignored_result = subprocess.run(["git", "-C", str(scratch), "check-ignore", "--no-index", "--stdin", "-z"],
                                        input=("\0".join(paths) + "\0").encode(), capture_output=True)
        assert ignored_result.returncode in (0, 1), ignored_result.stderr
        ignored = set(ignored_result.stdout.decode().strip("\0").split("\0"))
        assert set(probes).issubset(ignored)
        for local in ("AGENTS.md", "PROJECT_HANDOFF.md", "LEARNING_LOG.md"):
            assert local in ignored
        candidates = sorted(set(actual) - ignored)
        root_files = {"README.md", ".gitignore", ".gitattributes", "requirements.txt", "LICENSE",
                      "train_cifar10_vit.py", "train_cifar10_resnet18.py", *[path.name for path in ROOT.glob("unit*.py")]}
        for relative in candidates:
            assert relative in root_files or relative.startswith(("docs/", "scripts/", "results/")), relative
            assert (ROOT / relative).stat().st_size < 5 * 1024 * 1024, relative
            shutil.copyfile(ROOT / relative, scratch / relative)
        staged = subprocess.run(["git", "-C", str(scratch), "-c", "core.quotePath=false", "add", "--dry-run", "--", "."],
                                capture_output=True, text=True, check=True)
        assert len([line for line in staged.stdout.splitlines() if line.startswith("add ")]) == len(candidates)
    assert not scratch.exists()
    return {"scope": "temporary Git repository only; project Git state unchanged", "public_file_count": len(candidates),
            "public_bytes": sum((ROOT / name).stat().st_size for name in candidates),
            "public_files": candidates, "ignored_original_runs": True, "weight_and_cache_probes_ignored": True,
            "temporary_repository_cleaned": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-originals", action="store_true", help="需要本机原始 outputs 及任务 013 核查快照")
    parser.add_argument("--check-environment", action="store_true")
    parser.add_argument("--check-models", action="store_true", help="仅 CPU 初始化与结构/参数核对，不训练")
    parser.add_argument("--check-ignore", action="store_true", help="在临时 Git 仓库验证 ignore 和 add --dry-run")
    parser.add_argument("--compare-split", type=Path, help="将指定索引文件与公开原始索引逐字节比较")
    parser.add_argument("--report", type=Path, help="可选：向新的本地文件保存验收 JSON")
    args = parser.parse_args()
    report = {"artifacts": artifacts(args.check_originals), "documents": documents(),
              "training_executed": False, "official_test_evaluated": False}
    if args.compare_split:
        assert args.compare_split.read_bytes() == (ROOT / "results/tiny_vit/split_indices.json").read_bytes()
        report["provided_split_identical"] = True
    if args.check_environment or args.check_models:
        report["environment"] = environment(args.check_models)
    if args.check_ignore:
        report["git_rules"] = git_rules()
    report["passed"] = True
    encoded = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
