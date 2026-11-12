import json

import pytest

from experiments.lib.experiment import load_config
from experiments.lib.mqar_reporting import protocol_summary


def test_checkpoint_paths_resolve_under_the_supplied_root(tmp_path):
    config = {"parent_checkpoint": {"path": "run/selected.pth", "result_path": "run/result.json", "sha256": "pinned"},
              "warm_start_root": "reference", "cap": 24}
    recipe = tmp_path / "recipe.json"
    recipe.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="--checkpoint-root"):
        load_config(recipe)
    root = tmp_path / "external-results"
    loaded = load_config(recipe, root)
    assert loaded["parent_checkpoint"]["path"] == str(root / "run/selected.pth")
    assert loaded["parent_checkpoint"]["result_path"] == str(root / "run/result.json")
    assert loaded["parent_checkpoint"]["sha256"] == "pinned"
    assert loaded["warm_start_root"] == str(root / "reference")
    assert json.loads(recipe.read_text()) == config


def test_absolute_checkpoint_paths_do_not_need_a_root(tmp_path):
    checkpoint = tmp_path / "selected.pth"
    recipe = tmp_path / "recipe.json"
    recipe.write_text(json.dumps({"parent_checkpoints": [{"path": str(checkpoint)}]}))
    assert load_config(recipe)["parent_checkpoints"][0]["path"] == str(checkpoint)


def test_report_describes_the_actual_validation_budget():
    config = {"train_examples": 128, "validation_examples": 8, "test_examples": 8,
              "batch_size": 128, "epochs": 1, "vocab_size": 8192, "train_length": 256,
              "train_pairs": 16, "layers": 2, "dimension": 64, "ffn_dimension": 128,
              "seeds": [11, 22, 33], "workflow_validation": True}
    description = protocol_summary(config)
    assert "WORKFLOW VALIDATION ONLY" in description
    assert "128 sequences" in description
    assert "1 optimizer updates" in description
    assert "20,000" not in description
