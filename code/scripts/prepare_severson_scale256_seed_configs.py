from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import yaml


DEFAULT_PROCESSED_DIR = Path("/root/autodl-tmp/data/severson-2019/processed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Write the controlled Severson scale256 fast-encoder and rollout "
            "predictor configs used for seed reruns."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("encoder", "predictor"),
        required=True,
        help="Which config to write.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
        help="Processed Severson directory on the server.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Encoder seed for mode=encoder, predictor seed for mode=predictor.",
    )
    parser.add_argument(
        "--encoder-seed",
        type=int,
        default=42,
        help="Encoder seed label used in predictor run names.",
    )
    parser.add_argument(
        "--latent-cache",
        type=Path,
        default=None,
        help="Required for mode=predictor. Path to the 256-dim latent cache.",
    )
    parser.add_argument(
        "--run-tag",
        type=str,
        default=None,
        help="Optional run tag. Defaults to current timestamp.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/generated_configs"),
        help="Directory for generated YAML configs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_tag = args.run_tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "encoder":
        config, run_name = build_encoder_config(
            processed_dir=args.processed_dir,
            seed=args.seed,
            run_tag=run_tag,
        )
        out = args.output_dir / f"pretrain_encoder_severson_scale256_fast_seed{args.seed}_{run_tag}.yaml"
    else:
        if args.latent_cache is None:
            raise SystemExit("--latent-cache is required when --mode predictor.")
        config, run_name = build_predictor_config(
            processed_dir=args.processed_dir,
            encoder_seed=args.encoder_seed,
            predictor_seed=args.seed,
            latent_cache=args.latent_cache,
            run_tag=run_tag,
        )
        out = (
            args.output_dir
            / (
                "train_latent_tf_vbll_severson_scale256_newcfg_"
                f"encseed{args.encoder_seed}_predseed{args.seed}_{run_tag}.yaml"
            )
        )

    out.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(f"CONFIG={out}")
    print(f"RUN_NAME={run_name}")


def build_encoder_config(
    *,
    processed_dir: Path,
    seed: int,
    run_tag: str,
) -> tuple[dict, str]:
    split_file = processed_dir / "splits/cell_random_seed_42.jsonl"
    normalizer = processed_dir / "normalization/train_vit_seed_42.json"
    run_name = f"severson_scale256_fast_encoder_seed{seed}_{run_tag}"
    return (
        {
            "experiment": {
                "name": "severson_cycle_autoencoder",
                "seed": seed,
                "output_dir": "outputs/severson_cycle_autoencoder",
                "timestamped_output": True,
                "run_name": run_name,
            },
            "data": {
                "processed_dir": str(processed_dir),
                "split_file": str(split_file),
                "normalizer": str(normalizer),
                "channels": ["V", "I", "T"],
                "sequence_length": 1000,
            },
            "model": {
                "input_dim": 3,
                "latent_dim": 256,
                "d_model": 256,
                "num_layers": 4,
                "num_heads": 8,
            },
            "training": {
                "batch_size": 256,
                "num_workers": 8,
                "pin_memory": True,
                "persistent_workers": True,
                "prefetch_factor": 2,
                "log_every_n_steps": 50,
                "max_epochs": 120,
                "lr": 2.0e-4,
                "weight_decay": 1.0e-5,
                "scheduler": {
                    "enabled": True,
                    "name": "reduce_on_plateau",
                    "factor": 0.5,
                    "patience": 8,
                    "threshold": 1.0e-4,
                    "min_lr": 2.0e-5,
                },
                "early_stopping": {
                    "enabled": True,
                    "patience": 20,
                    "min_delta": 1.0e-5,
                },
                "device": "auto",
            },
            "logging": {
                "backend": "wandb",
                "project": "battery-soh-bnn",
                "offline": False,
            },
        },
        run_name,
    )


def build_predictor_config(
    *,
    processed_dir: Path,
    encoder_seed: int,
    predictor_seed: int,
    latent_cache: Path,
    run_tag: str,
) -> tuple[dict, str]:
    split_file = processed_dir / "splits/cell_random_seed_42.jsonl"
    if encoder_seed == 42:
        run_name = f"severson_fastenc_scale256_vbll_newcfg_predseed{predictor_seed}_{run_tag}"
    else:
        run_name = (
            "severson_fastenc_scale256_vbll_newcfg_"
            f"encseed{encoder_seed}_predseed{predictor_seed}_{run_tag}"
        )
    return (
        {
            "experiment": {
                "name": "severson_latent_tf_vbll",
                "seed": predictor_seed,
                "output_dir": "outputs/severson_latent_tf_vbll",
                "timestamped_output": True,
                "run_name": run_name,
            },
            "data": {
                "processed_dir": str(processed_dir),
                "split_file": str(split_file),
                "latent_cache": str(latent_cache),
                "window": 30,
                "horizon": 80,
                "latent_dim": 256,
                "aux_dim": 0,
            },
            "model": {
                "kind": "latent_tf_vbll",
                "history_dim": 512,
                "decoder_dim": 512,
                "vbll_prior_variance": 0.5,
            },
            "training": {
                "objective": "nll",
                "batch_size": 128,
                "num_workers": 8,
                "pin_memory": True,
                "persistent_workers": True,
                "prefetch_factor": 2,
                "log_every_n_steps": 50,
                "max_epochs": 120,
                "lr": 2.0e-4,
                "weight_decay": 1.0e-5,
                "beta_kl": 1.0e-4,
                "lambda_state": 1.0,
                "lambda_negative": 0.0,
                "lambda_smooth": 0.0,
                "device": "auto",
                "vbll_rho_lr_multiplier": 5.0,
                "scheduled_sampling": {
                    "enabled": True,
                    "start_epoch": 10,
                    "end_epoch": 40,
                    "start_probability": 0.0,
                    "end_probability": 0.25,
                    "validation_probability": 1.0,
                },
                "stochastic_rollout": {
                    "enabled": True,
                    "start_epoch": 10,
                    "prefix_mode": "generated",
                    "num_samples": 4,
                    "weight": 0.1,
                    "objective": "crps",
                    "negative_weight": 0.1,
                    "validation_num_samples": 8,
                    "validation_horizon": 80,
                    "validation_seed": 42,
                    "horizon_schedule": [
                        {"start_epoch": 10, "horizon": 20},
                        {"start_epoch": 30, "horizon": 40},
                        {"start_epoch": 50, "horizon": 80},
                    ],
                },
            },
            "logging": {
                "backend": "wandb",
                "project": "battery-soh-bnn",
                "offline": False,
            },
        },
        run_name,
    )


if __name__ == "__main__":
    main()
