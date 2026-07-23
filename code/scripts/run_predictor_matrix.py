from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cache encoder best/last latents and train VBLL/deterministic predictors."
    )
    parser.add_argument("--encoder-config", type=Path, required=True)
    parser.add_argument("--encoder-run-dir", type=Path, required=True)
    parser.add_argument(
        "--vbll-config",
        type=Path,
        default=Path("configs/train_latent_tf_vbll.yaml"),
    )
    parser.add_argument(
        "--deterministic-config",
        type=Path,
        default=Path("configs/train_latent_tf_deterministic.yaml"),
    )
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--run-prefix", type=str, default=None)
    parser.add_argument("--reuse-cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    code_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    src_path = str(code_root / "src")
    current_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{src_path}{os.pathsep}{current_pythonpath}" if current_pythonpath else src_path
    )

    encoder_run_dir = _resolve(code_root, args.encoder_run_dir)
    cache_root = _resolve(code_root, args.cache_root)
    run_prefix = args.run_prefix or encoder_run_dir.name
    variants = (
        ("vbll", args.vbll_config),
        ("deterministic", args.deterministic_config),
    )

    for checkpoint_name in ("best", "last"):
        checkpoint = encoder_run_dir / f"{checkpoint_name}.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Encoder checkpoint not found: {checkpoint}")
        latent_cache = cache_root / checkpoint_name / "latents.npz"
        if not (args.reuse_cache and latent_cache.is_file()):
            _run(
                [
                    sys.executable,
                    "scripts/cache_latents.py",
                    "--config",
                    str(args.encoder_config),
                    "--checkpoint",
                    str(checkpoint),
                    "--output",
                    str(latent_cache),
                ],
                cwd=code_root,
                env=env,
            )
        else:
            print(f"Reusing latent cache: {latent_cache}", flush=True)

        for variant_name, config in variants:
            run_name = f"{run_prefix}_{checkpoint_name}"
            print(
                f"Training {variant_name} with encoder checkpoint={checkpoint_name}",
                flush=True,
            )
            _run(
                [
                    sys.executable,
                    "scripts/train_predictor.py",
                    "--config",
                    str(config),
                    "--processed-dir",
                    str(args.processed_dir),
                    "--split-file",
                    str(args.split_file),
                    "--latent-cache",
                    str(latent_cache),
                    "--run-name",
                    run_name,
                ],
                cwd=code_root,
                env=env,
            )


def _resolve(code_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else code_root / path


def _run(command: list[str], cwd: Path, env: dict[str, str]) -> None:
    print(f"Running: {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


if __name__ == "__main__":
    main()
