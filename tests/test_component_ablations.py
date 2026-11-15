from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from battery_soh.models.soh_predictor import SOHProbabilisticPredictor
from battery_soh.training.trainer import build_predictor_optimizer, train_one_epoch

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load_script("run_revision_fair_rollout")
train = load_script("train_predictor")
REFERENCE = json.loads(runner.REFERENCE.read_text())["config"]


@pytest.mark.parametrize("variant,ss,sro", [
    ("no_ss", False, True), ("no_sro", True, False), ("neither", False, False),
])
def test_component_config_has_only_the_declared_deletion(variant, ss, sro):
    cfg = copy.deepcopy(REFERENCE)
    runner.apply_intervention(cfg, variant)
    runner.validate_controlled_config(cfg, REFERENCE, variant)
    assert cfg["training"]["scheduled_sampling"]["enabled"] is ss
    assert cfg["training"]["stochastic_rollout"]["enabled"] is sro
    assert cfg["training"]["stochastic_rollout"].get("crps_variant", "empirical") == "empirical"
    assert train.rollout_validation_settings(cfg) == train.rollout_validation_settings(REFERENCE)
    cfg["training"]["max_epochs"] = 60
    with pytest.raises(ValueError, match="differs beyond"):
        runner.validate_controlled_config(cfg, REFERENCE, variant)


def test_legacy_validation_behavior_is_unchanged_without_override():
    cfg = copy.deepcopy(REFERENCE)
    cfg["training"]["scheduled_sampling"]["enabled"] = False
    cfg["training"]["stochastic_rollout"]["enabled"] = False
    setting = train.rollout_validation_settings(cfg)
    assert setting["num_samples"] == 0
    assert setting["sampling_probability"] == 0


@pytest.mark.parametrize("variant,ss,sro", [
    ("no_ss", False, True), ("no_sro", True, False), ("neither", False, False),
])
def test_actual_training_switches_and_validation_are_independent(variant, ss, sro):
    cfg = copy.deepcopy(REFERENCE)
    runner.apply_intervention(cfg, variant)
    rollout = cfg["training"]["stochastic_rollout"]
    rollout.pop("horizon_schedule")
    rollout.update(start_epoch=1, horizon=3)
    cfg["training"]["scheduled_sampling"].update(start_epoch=1, end_epoch=2)
    model = SOHProbabilisticPredictor.vbll(
        latent_dim=4, aux_dim=0, horizon=3, history_dim=8, decoder_dim=8,
    )
    optimizer = build_predictor_optimizer(model, cfg)
    batch = {"z_window": torch.randn(2, 5, 4), "soh_history": torch.ones(2, 5),
             "soh_future": torch.full((2, 3), 0.9), "mask": torch.ones(2, 3, dtype=torch.bool)}
    trained = train_one_epoch(model, [batch], optimizer, cfg, torch.device("cpu"), epoch=2)
    assert trained.metrics["scheduled_sampling_probability"] == (0.25 if ss else 0)
    assert trained.metrics["stochastic_rollout_horizon"] == (3 if sro else 0)
    if not sro:
        assert trained.metrics["stochastic_rollout"] == 0
        assert trained.metrics["stochastic_rollout_negative"] == 0
    optimizer.zero_grad(set_to_none=True)
    before_rng = torch.random.get_rng_state().clone()
    validated = train._evaluate_rollout_validation(
        model, [batch], cfg, torch.device("cpu"), probability=1.0,
        stochastic_num_samples=8, stochastic_horizon=3,
        stochastic_prefix_mode="generated", seed=42,
    )
    assert validated.metrics["stochastic_rollout_horizon"] == 3
    assert validated.metrics["stochastic_rollout"] >= 0
    assert all(p.grad is None for p in model.parameters())
    assert torch.equal(before_rng, torch.random.get_rng_state())
