from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_CELLS = ("CS2_35", "CS2_36", "CS2_37", "CS2_38")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run CALCE CS2 1C leave-one-cell-out experiments end to end: "
            "split, train-only normalizer, encoder, latent cache, predictor, "
            "best/last rollout evaluation, and aggregate report."
        )
    )
    parser.add_argument("--processed-dir", type=Path, default=Path("~/data/calce/processed"))
    parser.add_argument("--pretrain-config", type=Path, default=Path("configs/pretrain_encoder_calce.yaml"))
    parser.add_argument("--predictor-config", type=Path, default=Path("configs/train_latent_tf_vbll_calce.yaml"))
    parser.add_argument("--cells", nargs="+", default=list(DEFAULT_CELLS))
    parser.add_argument(
        "--folds",
        nargs="+",
        default=None,
        help="Optional subset of test cells to run. Default: every value in --cells.",
    )
    parser.add_argument(
        "--val-policy",
        choices=("previous", "next"),
        default="previous",
        help="Choose validation cell adjacent to the held-out test cell in sorted cell order.",
    )
    parser.add_argument("--run-prefix", default=None)
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=1000)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    processed_dir = args.processed_dir.expanduser()
    run_prefix = args.run_prefix or f"calce_cs2_1c_loo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    cells = list(args.cells)
    folds = list(args.folds) if args.folds else cells
    _validate_cells(cells, folds)

    artifact_root = processed_dir / "loo_1c" / run_prefix
    splits_dir = artifact_root / "splits"
    normalizer_dir = artifact_root / "normalization"
    config_dir = artifact_root / "configs"
    cache_root = processed_dir / "latent_cache" / run_prefix
    analysis_dir = root / "outputs" / "analysis" / run_prefix
    for path in (splits_dir, normalizer_dir, config_dir, cache_root, analysis_dir):
        path.mkdir(parents=True, exist_ok=True)

    print("=== CALCE 1C leave-one-cell-out ===", flush=True)
    print(f"run_prefix={run_prefix}", flush=True)
    print(f"processed_dir={processed_dir}", flush=True)
    print(f"cells={cells}", flush=True)
    print(f"folds={folds}", flush=True)
    print(f"artifact_root={artifact_root}", flush=True)
    print(f"analysis_dir={analysis_dir}", flush=True)

    fold_rows: list[dict[str, Any]] = []
    for test_cell in folds:
        fold = _make_fold(cells, test_cell, args.val_policy)
        fold_name = f"test_{test_cell}"
        print(f"\n=== Fold {fold_name} ===", flush=True)
        print(json.dumps(fold, ensure_ascii=False, sort_keys=True), flush=True)
        fold_rows.append(
            run_fold(
                root=root,
                processed_dir=processed_dir,
                artifact_root=artifact_root,
                splits_dir=splits_dir,
                normalizer_dir=normalizer_dir,
                config_dir=config_dir,
                cache_root=cache_root,
                pretrain_config=args.pretrain_config,
                predictor_config=args.predictor_config,
                run_prefix=run_prefix,
                fold_name=fold_name,
                fold=fold,
                sequence_length=args.sequence_length,
                num_samples=args.num_samples,
                eval_batch_size=args.eval_batch_size,
                num_workers=args.num_workers,
                skip_existing=args.skip_existing,
            )
        )
        write_reports(analysis_dir, fold_rows)

    print("\n=== Aggregated results ===", flush=True)
    print_table(fold_rows)
    write_reports(analysis_dir, fold_rows)
    print(f"Wrote aggregate reports to {analysis_dir}", flush=True)


def run_fold(
    *,
    root: Path,
    processed_dir: Path,
    artifact_root: Path,
    splits_dir: Path,
    normalizer_dir: Path,
    config_dir: Path,
    cache_root: Path,
    pretrain_config: Path,
    predictor_config: Path,
    run_prefix: str,
    fold_name: str,
    fold: dict[str, list[str]],
    sequence_length: int,
    num_samples: int,
    eval_batch_size: int,
    num_workers: int,
    skip_existing: bool,
) -> dict[str, Any]:
    split_file = splits_dir / f"{fold_name}.jsonl"
    normalizer = normalizer_dir / f"{fold_name}_train_vi.json"
    pretrain_cfg_path = config_dir / f"pretrain_encoder_calce_1c_loo_{fold_name}.yaml"
    predictor_cfg_path = config_dir / f"train_latent_tf_vbll_calce_1c_loo_{fold_name}.yaml"

    encoder_run_name = f"{run_prefix}_{fold_name}_encoder"
    predictor_run_name = f"{run_prefix}_{fold_name}_vbll"
    write_jsonl(split_file, [fold])

    run_cmd(
        [
            sys.executable,
            "-u",
            "scripts/fit_channel_normalizer.py",
            "--processed-dir",
            str(processed_dir),
            "--split-file",
            str(split_file),
            "--split",
            "train",
            "--channels",
            "V",
            "I",
            "--sequence-length",
            str(sequence_length),
            "--output",
            str(normalizer),
        ],
        cwd=root,
        skip=skip_existing and normalizer.is_file(),
        label=f"{fold_name}: normalizer",
    )

    pretrain_cfg = load_yaml(pretrain_config)
    pretrain_cfg.setdefault("data", {})["processed_dir"] = str(processed_dir)
    pretrain_cfg["data"]["split_file"] = str(split_file)
    pretrain_cfg["data"]["normalizer"] = str(normalizer)
    pretrain_cfg["data"]["channels"] = ["V", "I"]
    pretrain_cfg["data"]["sequence_length"] = sequence_length
    pretrain_cfg.setdefault("model", {})["input_dim"] = 2
    pretrain_cfg.setdefault("experiment", {})["run_name"] = encoder_run_name
    dump_yaml(pretrain_cfg_path, pretrain_cfg)

    predictor_cfg = load_yaml(predictor_config)
    predictor_cfg.setdefault("data", {})["processed_dir"] = str(processed_dir)
    predictor_cfg["data"]["split_file"] = str(split_file)
    predictor_cfg.setdefault("experiment", {})["run_name"] = predictor_run_name
    dump_yaml(predictor_cfg_path, predictor_cfg)

    encoder_run = run_dir_from_config(pretrain_cfg)
    checkpoint = encoder_run / "best.pt"
    run_cmd(
        [
            sys.executable,
            "-u",
            "scripts/pretrain_encoder.py",
            "--config",
            str(pretrain_cfg_path),
        ],
        cwd=root,
        skip=skip_existing and checkpoint.is_file(),
        label=f"{fold_name}: encoder",
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Encoder checkpoint not found: {checkpoint}")

    cache_dir = cache_root / f"{encoder_run_name}_best"
    cache = cache_dir / "latents.npz"
    cache_dir.mkdir(parents=True, exist_ok=True)
    run_cmd(
        [
            sys.executable,
            "-u",
            "scripts/cache_latents.py",
            "--config",
            str(pretrain_cfg_path),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(cache),
        ],
        cwd=root,
        skip=skip_existing and cache.is_file(),
        label=f"{fold_name}: cache",
    )

    predictor_run = run_dir_from_config(predictor_cfg)
    predictor_best = predictor_run / "best.pt"
    run_cmd(
        [
            sys.executable,
            "-u",
            "scripts/train_predictor.py",
            "--config",
            str(predictor_cfg_path),
            "--split-file",
            str(split_file),
            "--latent-cache",
            str(cache),
            "--run-name",
            predictor_run_name,
        ],
        cwd=root,
        skip=skip_existing and predictor_best.is_file(),
        label=f"{fold_name}: predictor",
    )
    if not predictor_best.is_file():
        raise FileNotFoundError(f"Predictor checkpoint not found: {predictor_best}")

    for checkpoint_name in ("best", "last"):
        checkpoint_path = predictor_run / f"{checkpoint_name}.pt"
        output_dir = predictor_run / f"rollout_test_{checkpoint_name}_samples_{num_samples}"
        run_cmd(
            [
                sys.executable,
                "-u",
                "scripts/evaluate.py",
                "--checkpoint",
                str(checkpoint_path),
                "--split",
                "test",
                "--latent-cache",
                str(cache),
                "--split-file",
                str(split_file),
                "--num-samples",
                str(num_samples),
                "--batch-size",
                str(eval_batch_size),
                "--num-workers",
                str(num_workers),
                "--save-samples",
                "--output-dir",
                str(output_dir),
            ],
            cwd=root,
            skip=skip_existing and (output_dir / "metrics.json").is_file(),
            label=f"{fold_name}: evaluate {checkpoint_name}",
        )

    row = {
        "fold": fold_name,
        "train_cells": ",".join(fold["train"]),
        "val_cell": ",".join(fold["val"]),
        "test_cell": ",".join(fold["test"]),
        "split_file": str(split_file),
        "normalizer": str(normalizer),
        "encoder_run": str(encoder_run),
        "latent_cache": str(cache),
        "predictor_run": str(predictor_run),
    }
    for checkpoint_name in ("best", "last"):
        row.update(read_eval_summary(predictor_run / f"rollout_test_{checkpoint_name}_samples_{num_samples}", checkpoint_name))
    return row


def read_eval_summary(output_dir: Path, prefix: str) -> dict[str, Any]:
    import numpy as np

    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    data = np.load(output_dir / "predictions.npz", allow_pickle=False)
    try:
        mask = data["mask"].astype(bool)
        lower_90 = data["lower_90"]
        lower90_neg = float(np.mean(lower_90[mask] < 0.0)) if np.any(mask) else float("nan")
        sample_neg = float("nan")
        sample_q01 = float("nan")
        if "samples" in data.files:
            samples = data["samples"]
            valid = np.broadcast_to(mask[None, :, :], samples.shape)
            values = samples[valid]
            values = values[np.isfinite(values)]
            if values.size:
                sample_neg = float(np.mean(values < 0.0))
                sample_q01 = float(np.quantile(values, 0.01))
    finally:
        data.close()

    values = metrics["metrics"]
    intervals = values.get("intervals", {})
    return {
        f"{prefix}_epoch": checkpoint_epoch(metrics["checkpoint"]),
        f"{prefix}_mae": values["overall"]["mae"],
        f"{prefix}_rmse": values["overall"]["rmse"],
        f"{prefix}_crps": values.get("crps", float("nan")),
        f"{prefix}_cov50": intervals.get("50", {}).get("coverage", float("nan")),
        f"{prefix}_wid50": intervals.get("50", {}).get("mean_width", float("nan")),
        f"{prefix}_cov90": intervals.get("90", {}).get("coverage", float("nan")),
        f"{prefix}_wid90": intervals.get("90", {}).get("mean_width", float("nan")),
        f"{prefix}_lower90_neg": lower90_neg,
        f"{prefix}_sample_neg": sample_neg,
        f"{prefix}_sample_q01": sample_q01,
    }


def checkpoint_epoch(checkpoint: str) -> int | str:
    try:
        import torch

        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        return int(payload.get("epoch"))
    except Exception:
        return ""


def write_reports(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "calce_1c_leave_one_out_summary.json"
    csv_path = output_dir / "calce_1c_leave_one_out_summary.csv"
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_table(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    print(
        f"{'fold':<14} {'test':<8} {'best_MAE':>9} {'best_CRPS':>10} "
        f"{'best_cov90':>11} {'best_wid90':>11} {'best_neg%':>10} "
        f"{'last_MAE':>9} {'last_CRPS':>10}",
        flush=True,
    )
    for row in rows:
        print(
            f"{row['fold']:<14} {row['test_cell']:<8} "
            f"{row['best_mae']:9.5f} {row['best_crps']:10.5f} "
            f"{row['best_cov90']:11.5f} {row['best_wid90']:11.5f} "
            f"{100.0 * row['best_sample_neg']:10.3f} "
            f"{row['last_mae']:9.5f} {row['last_crps']:10.5f}",
            flush=True,
        )


def _make_fold(cells: list[str], test_cell: str, val_policy: str) -> dict[str, list[str]]:
    index = cells.index(test_cell)
    if val_policy == "previous":
        val_cell = cells[index - 1]
    else:
        val_cell = cells[(index + 1) % len(cells)]
    train = [cell for cell in cells if cell not in {test_cell, val_cell}]
    return {"train": train, "val": [val_cell], "test": [test_cell]}


def _validate_cells(cells: list[str], folds: list[str]) -> None:
    if len(cells) < 3:
        raise ValueError("At least three cells are required for train/val/test folds.")
    if len(set(cells)) != len(cells):
        raise ValueError(f"Duplicate cells are not allowed: {cells}")
    missing = [cell for cell in folds if cell not in cells]
    if missing:
        raise ValueError(f"Fold cells not found in --cells: {missing}")


def run_cmd(command: list[str], *, cwd: Path, skip: bool, label: str) -> None:
    if skip:
        print(f"--- SKIP {label}", flush=True)
        return
    print(f"--- RUN {label}", flush=True)
    print(" ".join(command), flush=True)
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = "src" if not existing else f"src{os.pathsep}{existing}"
    subprocess.run(command, cwd=cwd, env=env, check=True)


def run_dir_from_config(cfg: dict[str, Any]) -> Path:
    experiment = cfg.get("experiment", {})
    base = Path(experiment["output_dir"])
    if not bool(experiment.get("timestamped_output", True)):
        return base
    return base / "runs" / str(experiment["run_name"])


def load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.expanduser().open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dump_yaml(path: Path, cfg: dict[str, Any]) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            f.write("\n")


if __name__ == "__main__":
    main()
