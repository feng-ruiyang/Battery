from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "revision_runner", ROOT / "scripts/run_revision_fair_rollout.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
REFERENCE = json.loads(
    (ROOT / "configs/revision/rollout_main_seed42_reference.json").read_text()
)["config"]


def candidate():
    cfg = copy.deepcopy(REFERENCE)
    cfg["training"]["stochastic_rollout"]["crps_variant"] = "fair"
    return cfg


def test_only_estimator_changes_scientifically():
    cfg = candidate()
    cfg["data"]["latent_cache"] = "/new-machine/latents.npz"
    cfg["data"]["split_file"] = "/new-machine/split.jsonl"
    cfg["experiment"]["output_dir"] = "/shared/revision"
    cfg["experiment"]["run_name"] = "controlled_fair"
    cfg["logging"] = {"backend": "csv"}
    runner.validate_controlled_config(cfg, REFERENCE)


@pytest.mark.parametrize("path,value", [
    (("training", "batch_size"), 64),
    (("model", "history_dim"), 256),
    (("training", "lr"), 1e-4),
    (("training", "stochastic_rollout", "weight"), 1.0),
    (("training", "stochastic_rollout", "validation_num_samples"), 0),
    (("training", "scheduled_sampling", "max_probability"), 0.25),
    (("training", "stochastic_rollout", "crps_variant"), "empirical"),
    (("experiment", "seed"), 8209),
])
def test_rejects_the_previous_confounds(path, value):
    cfg = candidate()
    node = cfg
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    with pytest.raises(ValueError, match="differs beyond"):
        runner.validate_controlled_config(cfg, REFERENCE)


def test_rejects_missing_curriculum():
    cfg = candidate()
    cfg["training"]["stochastic_rollout"].pop("horizon_schedule")
    cfg["training"]["stochastic_rollout"]["horizon"] = 80
    with pytest.raises(ValueError, match="differs beyond"):
        runner.validate_controlled_config(cfg, REFERENCE)
