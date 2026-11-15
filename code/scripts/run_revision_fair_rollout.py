"""Prepare, verify and run controlled CRPS-estimator or component ablations.

The empirical checkpoint supplies the reference configuration.
Each variant starts training from seed 42.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

CODE = Path(__file__).resolve().parents[1]
REFERENCE = CODE / "configs/revision/rollout_main_seed42_reference.json"
METHODS = {
    "fair": "VBLL-Rollout-CRPS-fair",
    "no_ss": "VBLL-Rollout-CRPS-w/o-SS",
    "no_sro": "VBLL-Rollout-CRPS-w/o-SRO",
    "neither": "VBLL-Rollout-CRPS-w/o-SS-and-SRO",
}
COMMON_VALIDATION = {"sampling_probability": 1.0, "num_samples": 8,
                     "horizon": 80, "seed": 42, "prefix_mode": "generated"}


def apply_intervention(cfg: dict, variant: str) -> None:
    if variant not in METHODS:
        raise ValueError(f"Unknown intervention: {variant}")
    if variant == "fair":
        cfg["training"]["stochastic_rollout"]["crps_variant"] = "fair"
    else:
        if variant in {"no_ss", "neither"}:
            cfg["training"]["scheduled_sampling"]["enabled"] = False
        if variant in {"no_sro", "neither"}:
            cfg["training"]["stochastic_rollout"]["enabled"] = False
        cfg["training"]["rollout_validation"] = copy.deepcopy(COMMON_VALIDATION)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".updating")
    temp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def scientific_config(cfg: dict) -> dict:
    result = copy.deepcopy(cfg)
    for name in ("processed_dir", "latent_cache", "split_file"):
        result["data"].pop(name, None)
    for name in ("output_dir", "run_name"):
        result["experiment"].pop(name, None)
    result.pop("logging", None)
    return result


def validate_controlled_config(cfg: dict, baseline: dict, variant: str = "fair") -> None:
    expected = scientific_config(baseline)
    apply_intervention(expected, variant)
    if scientific_config(cfg) != expected:
        raise ValueError(f"Scientific configuration differs beyond the declared {variant} intervention")


def source_hashes() -> dict[str, str]:
    paths = []
    for folder in ("src", "scripts", "configs"):
        paths.extend(p for p in (CODE / folder).rglob("*")
                     if p.is_file() and p.suffix in {".py", ".json", ".yaml", ".yml"})
    return {str(p.relative_to(CODE)): sha256(p) for p in sorted(paths)}


def verified_copy(src: Path, dst: Path, expected: str | None = None) -> dict:
    source_digest = sha256(src)
    if expected is not None and source_digest != expected:
        raise ValueError(f"Audited checkpoint hash mismatch: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if sha256(dst) != source_digest:
            raise ValueError(f"Refusing to replace a different input: {dst}")
    else:
        temp = dst.with_suffix(dst.suffix + ".copying")
        shutil.copy2(src, temp)
        if sha256(temp) != source_digest:
            raise ValueError(f"Input copy verification failed: {dst}")
        temp.replace(dst)
    return {"source": str(src), "path": str(dst), "sha256": source_digest}


def prepare(args: argparse.Namespace) -> None:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this experiment")
    ref = json.loads(REFERENCE.read_text())
    shared = args.shared_root.resolve()
    if not shared.is_dir():
        raise FileNotFoundError(shared)
    inputs = args.input_dir.resolve()
    records = {
        "baseline": verified_copy(shared / ref["checkpoint_relative"],
                                  inputs / "baseline_main_seed42_epoch94.pt",
                                  ref["checkpoint_sha256"]),
        "cache": verified_copy(shared / ref["cache_relative"],
                               inputs / "fast_encoder_20260808_002315_best/latents.npz"),
        "split": verified_copy(shared / ref["split_relative"],
                               inputs / "cell_random_seed_42.jsonl"),
    }
    payload = torch.load(records["baseline"]["path"], map_location="cpu", weights_only=True)
    if payload["config"] != ref["config"] or payload["epoch"] != ref["best_epoch"]:
        raise ValueError("Reference checkpoint metadata does not match the reference configuration")
    cfg = copy.deepcopy(payload["config"])
    del payload
    variant = args.variant
    name = f"severson_fastenc_scale256_vbll_{variant}_controlled_seed42_" + datetime.now(
        ZoneInfo("Asia/Shanghai")
    ).strftime("%Y%m%d_%H%M%S")
    output_root = shared / ("revision_fair_rollout" if variant == "fair" else "revision_component_ablation")
    run = output_root / "runs" / name
    run.mkdir(parents=True, exist_ok=False)
    cfg["experiment"].update(output_dir=str(output_root), run_name=name)
    cfg["data"].update(processed_dir=str(shared / "data/severson-2019/processed"),
                       latent_cache=records["cache"]["path"], split_file=records["split"]["path"])
    apply_intervention(cfg, variant)
    cfg["logging"] = {"backend": "csv"}
    validate_controlled_config(cfg, ref["config"], variant)
    write_json(run / "config.json", cfg)  # JSON is accepted by the YAML config loader.
    hashes = source_hashes()
    for rel in hashes:
        dst = run / "provenance/source" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(CODE / rel, dst)
    write_json(run / "provenance/reference.json", ref)
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                            capture_output=True, text=True, check=True)
    (run / "provenance/pip-freeze.txt").write_text(freeze.stdout)
    manifest = {
        "method": METHODS[variant], "variant": variant, "run_dir": str(run),
        "code_dir": str(CODE), "python": sys.executable, "inputs": records,
        "config_sha256": sha256(run / "config.json"), "source_sha256": hashes,
        "scientific_change": {
            "fair": {"training.stochastic_rollout.crps_variant": ["empirical", "fair"]},
            "no_ss": {"training.scheduled_sampling.enabled": [True, False]},
            "no_sro": {"training.stochastic_rollout.enabled": [True, False]},
            "neither": {"training.scheduled_sampling.enabled": [True, False],
                        "training.stochastic_rollout.enabled": [True, False]},
        }[variant],
        "selection": "generated-prefix validation CRPS + 0.1 * negative loss; K8 H80 seed42",
        "infrastructure_changes": ["input/output paths", "run name", "CSV logging instead of W&B"],
        "initialization": "fresh seed 42; reference weights are NOT used for training",
        "evaluation": {"split": "test", "num_samples": 128, "batch_size": 32,
                       "num_workers": 8, "seed": 42, "save_samples": True},
        "environment": {"python": sys.version, "torch": torch.__version__,
                        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
                        "host": platform.node()},
    }
    write_json(run / "manifest.json", manifest)
    print(json.dumps({"run_dir": str(run), "config": str(run / "config.json")}), flush=True)


def verify_run(run: Path) -> tuple[dict, dict]:
    manifest = json.loads((run / "manifest.json").read_text())
    cfg = json.loads((run / "config.json").read_text())
    ref = json.loads((run / "provenance/reference.json").read_text())
    validate_controlled_config(cfg, ref["config"], manifest.get("variant", "fair"))
    if sha256(run / "config.json") != manifest["config_sha256"]:
        raise ValueError("Prepared configuration was modified")
    if source_hashes() != manifest["source_sha256"]:
        raise ValueError("Code changed after preparation; create a new prepared run")
    for item in manifest["inputs"].values():
        if sha256(Path(item["path"])) != item["sha256"]:
            raise ValueError(f"Input changed: {item['path']}")
    return cfg, manifest


def preflight(run: Path) -> None:
    import torch
    from torch.utils.data import default_collate
    from battery_soh.data.torch_datasets import LatentCacheStore, LatentWindowDataset, load_split_ids
    from battery_soh.training.losses import stochastic_rollout_loss, total_loss
    from battery_soh.training.trainer import (
        build_predictor_optimizer, stochastic_rollout_settings, scheduled_sampling_probability,
    )
    from battery_soh.utils.seed import seed_everything
    from train_predictor import build_model, rollout_validation_settings
    cfg, manifest = verify_run(run)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    # Check the CRPS variant selected by the training objective.
    samples = torch.tensor([0., 2., 4., 6.]).reshape(4, 1, 1)
    batch = {"soh_future": torch.tensor([[1.]]), "mask": torch.ones(1, 1, dtype=torch.bool)}
    fair = copy.deepcopy(cfg)
    fair["training"]["stochastic_rollout"]["crps_variant"] = "fair"
    empirical = copy.deepcopy(cfg)
    empirical["training"]["stochastic_rollout"]["crps_variant"] = "empirical"
    outputs = {"stochastic_rollout_samples": samples}
    torch.testing.assert_close(stochastic_rollout_loss(outputs, batch, fair), torch.tensor(5 / 6))
    torch.testing.assert_close(stochastic_rollout_loss(outputs, batch, empirical), torch.tensor(1.25))
    for epoch, expected in [(9, (0, 0)), (10, (4, 20)), (30, (4, 40)), (50, (4, 80))]:
        if not cfg["training"]["stochastic_rollout"]["enabled"]:
            expected = (0, 0)
        if stochastic_rollout_settings(cfg, epoch) != expected:
            raise ValueError(f"Unexpected rollout curriculum at epoch {epoch}")
    cache = LatentCacheStore(cfg["data"]["latent_cache"])
    if cache.z.shape != (114314, 256):
        raise ValueError(f"Unexpected latent shape: {cache.z.shape}")
    ids = {s: load_split_ids(cfg["data"]["split_file"], s) or [] for s in ("train", "val", "test")}
    if [len(ids[s]) for s in ids] != [98, 21, 21]:
        raise ValueError("Unexpected cell split counts")
    if len(set(ids["train"] + ids["val"] + ids["test"])) != 140:
        raise ValueError("Overlapping cell split")
    datasets = {s: LatentWindowDataset(cache, cells, 30, 80) for s, cells in ids.items()}
    if len(datasets["test"]) != 17317:
        raise ValueError("Test windows differ from the reference experiment")
    items = []
    for index in range(len(datasets["train"])):
        item = datasets["train"][index]
        if bool(item["mask"].all()):
            items.append(item)
        if len(items) == 128:
            break
    if len(items) != 128:
        raise ValueError("Cannot form a complete-horizon smoke-test batch")
    batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v
             for k, v in default_collate(items).items()}
    seed_everything(42)
    model = build_model(cfg).cuda().train()
    optimizer = build_predictor_optimizer(model, cfg)
    torch.cuda.reset_peak_memory_stats()
    ss = scheduled_sampling_probability(cfg, 50)
    count, horizon = stochastic_rollout_settings(cfg, 50)
    validation = rollout_validation_settings(cfg)
    if validation != COMMON_VALIDATION:
        raise ValueError("Checkpoint selection must match the complete model")
    print(f"GPU preflight: batch128, rollout H{horizon}, K{count}, SS={ss}", flush=True)
    outputs = model(z_window=batch["z_window"], soh_history=batch["soh_history"],
                    soh_future=batch["soh_future"], history_mask=batch["history_mask"],
                    scheduled_sampling_probability=ss, stochastic_rollout_num_samples=count,
                    stochastic_rollout_horizon=horizon, stochastic_rollout_prefix_mode="generated")
    losses = total_loss(outputs, batch, cfg)
    if count == 0 and ("stochastic_rollout_samples" in outputs
                       or float(losses["stochastic_rollout"]) != 0
                       or float(losses["stochastic_rollout_negative"]) != 0):
        raise ValueError("Disabled SRO still contributes to training")
    if not torch.isfinite(losses["loss"]):
        raise ValueError("Non-finite preflight loss")
    losses["loss"].backward()
    if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
        raise ValueError("Non-finite preflight gradient")
    optimizer.step()
    torch.cuda.synchronize()
    loss_value = float(losses["loss"].detach())
    rollout_value = float(losses["stochastic_rollout"].detach())
    del outputs, losses
    optimizer.zero_grad(set_to_none=True)
    model.eval()
    with torch.no_grad():
        val_outputs = model(z_window=batch["z_window"], soh_history=batch["soh_history"],
                            soh_future=batch["soh_future"], history_mask=batch["history_mask"],
                            scheduled_sampling_probability=1.0, stochastic_rollout_num_samples=8,
                            stochastic_rollout_horizon=80, stochastic_rollout_prefix_mode="generated")
        val_loss = stochastic_rollout_loss(val_outputs, batch, cfg)
    if not bool(torch.isfinite(val_loss)) or any(p.grad is not None for p in model.parameters()):
        raise ValueError("Invalid validation loss or validation updated gradients")
    result = {"passed": True, "config_sha256": manifest["config_sha256"],
              "windows": {s: len(d) for s, d in datasets.items()},
              "loss": loss_value, "stochastic_rollout_loss": rollout_value,
              "training_ss_probability": ss, "training_samples": count,
              "training_rollout_horizon": horizon, "validation": validation,
              "validation_crps": float(val_loss),
              "peak_cuda_memory_gib": torch.cuda.max_memory_allocated() / 1024**3,
              "peak_cuda_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3}
    write_json(run / "preflight.json", result)
    print(json.dumps(result), flush=True)


def run_experiment(run: Path) -> None:
    import torch
    cfg, manifest = verify_run(run)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    checked = json.loads((run / "preflight.json").read_text())
    if not checked["passed"] or checked["config_sha256"] != manifest["config_sha256"]:
        raise ValueError("A matching successful preflight is required")
    if (run / "best.pt").exists() or (run / "last.pt").exists():
        raise FileExistsError("Existing training checkpoints: refuse to overwrite/restart")
    with (run / "run.lock").open("x") as f:
        json.dump({"pid": os.getpid(), "host": platform.node()}, f)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(CODE / "src")
    logs = run / "logs"
    logs.mkdir(exist_ok=True)

    def execute(stage: str, command: list[str]) -> None:
        write_json(run / "state.json", {"stage": stage, "status": "running", "command": command})
        print(f"Starting {stage}: {' '.join(command)}", flush=True)
        with (logs / (stage + ".log")).open("x") as output:
            subprocess.run(command, cwd=CODE, env=env, stdout=output,
                           stderr=subprocess.STDOUT, check=True)

    try:
        execute("training", [sys.executable, "-u", str(CODE / "scripts/train_predictor.py"),
                             "--config", str(run / "config.json")])
        variant = manifest.get("variant", "fair")
        jobs = [(f"{variant}_best_samples128", run / "best.pt"),
                (f"{variant}_last_samples128", run / "last.pt")]
        if variant == "fair":
            jobs.append(("baseline_best_samples128", Path(manifest["inputs"]["baseline"]["path"])))
        summary = {}
        for label, checkpoint in jobs:
            output = run / "evaluations" / label
            command = [sys.executable, "-u", str(CODE / "scripts/evaluate.py"),
                       "--checkpoint", str(checkpoint), "--latent-cache", cfg["data"]["latent_cache"],
                       "--split-file", cfg["data"]["split_file"], "--split", "test",
                       "--num-samples", "128", "--batch-size", "32", "--num-workers", "8",
                       "--seed", "42", "--save-samples", "--output-dir", str(output)]
            execute(label, command)
            write_json(output / "provenance.json", {"command": command,
                       "checkpoint_sha256": sha256(checkpoint), "inputs": manifest["inputs"],
                       "source_sha256": manifest["source_sha256"], "evaluation": manifest["evaluation"]})
            summary[label] = json.loads((output / "metrics.json").read_text())
        write_json(run / "evaluation_summary.json", summary)
        write_json(run / "state.json", {"stage": "complete", "status": "complete"})
        print("Training and all planned evaluations completed on shared storage", flush=True)
    except BaseException as error:
        write_json(run / "state.json", {"stage": "failed", "status": "failed", "error": repr(error)})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--variant", choices=tuple(METHODS), default="fair")
    p.add_argument("--shared-root", type=Path, default=Path("/root/autodl-fs/BatteryProject"))
    p.add_argument("--input-dir", type=Path, default=Path("/root/autodl-tmp/battery_revision_inputs"))
    for action in ("preflight", "run"):
        p = sub.add_parser(action)
        p.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args)
    elif args.action == "preflight":
        preflight(args.run_dir.resolve())
    else:
        run_experiment(args.run_dir.resolve())


if __name__ == "__main__":
    main()
