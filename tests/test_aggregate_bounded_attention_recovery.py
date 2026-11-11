import copy
import json
import math

import pytest

from quantum_friendly_transformer.experiments.lib.retained import RETAINED_BOUNDED_ATTENTION_RECOVERY_RUNS
from quantum_friendly_transformer.experiments.runners.reports.aggregate_bounded_attention_recovery import build_row


EXPECTED_SOURCES = {
    "raw/recovery_sst2_tinystories_1m_fn24_bounded_alpha_attention_steps20000.json",
    "raw/recovery_sst2_tinystories_3m_fn24_bounded_alpha_attention_steps20000.json",
    "raw/recovery_sst2_tinystories_8m_fn24_bounded_alpha_attention_shared_scalelr3e2_steps5000.json",
    "raw/recovery_sst2_tinystories_28m_fn24_bounded_alpha_attention_steps10000.json",
    "raw/recovery_sst2_tinystories_33m_fn24_bounded_alpha_attention_shared_steps3000.json",
    "raw/recovery_sst2_tinystories_1layer_21m_fn24_bounded_alpha_attention_steps10000.json",
    "raw/recovery_sst2_tinystories_2layers_33m_fn24_bounded_alpha_attention_shared_steps5000.json",
    "raw/recovery_sst2_olmo_1b_fn24_bounded_alpha_attention_shared_steps6315.json",
    "raw/recovery_sst2_mistral7b_fn24_bounded_alpha_attention_shared_steps500.json",
}


def _scale_stat(scale: float) -> dict:
    ceiling = 0.025
    floor = 40.0
    return {
        "name": "attention",
        "shared": False,
        "controller": "AlphaBoundedAttentionScale",
        "initial_score_scale": ceiling,
        "score_scale": scale,
        "score_denominator": 1.0 / scale,
        "effective_alpha_product": 1.0 / math.sqrt(scale),
        "log_scale": math.log(scale),
        "min_score_scale": 1e-12,
        "max_score_scale": ceiling,
        "score_scale_ceiling": ceiling,
        "score_denominator_floor": floor,
        "alpha_denominator_bound_enforced": True,
        "scale_to_alpha_ratio": scale / ceiling,
        "denominator_to_alpha_ratio": ceiling / scale,
        "constraint_active": scale == ceiling,
        "projection_count": 0,
        "raw_scale": None,
        "log_denominator_slack": math.log(ceiling / scale),
    }


def _artifact() -> dict:
    initial = _scale_stat(0.025)
    final = _scale_stat(0.0125)
    return {
        "model": "example/model",
        "task": "sst2",
        "caps": [24],
        "baseline_target": 0.81,
        "threshold": 0.80,
        "attention_scaling": "alpha_s_alpha_w",
        "trainable_attention_scaling": True,
        "shared_attention_scale": False,
        "enforce_alpha_denominator_bound": True,
        "attention_scale_native_regularization": 0.0,
        "attention_scale_prior_warmup_steps": 0,
        "attention_scale_prior_warmup_snap_native": False,
        "alpha_s": 2.0,
        "alpha_w": 3.0,
        "results": [
            {
                "cap": 24.0,
                "applied_caps": 6,
                "alpha_denominator_bound_enforced": True,
                "attention_scale_frozen_after_warmup": False,
                "effective_alpha_s": 2.0,
                "attention_scaling_layers": [
                    {
                        "name": "attention",
                        "scaling": "alpha_s_alpha_w",
                        "trainable": True,
                        "enforce_alpha_denominator_bound": True,
                        "alpha_s": 2.0,
                        "alpha_w": 3.0,
                        "score_scale": 0.025,
                        "score_denominator": 40.0,
                        "score_scale_ceiling": 0.025,
                        "score_denominator_floor": 40.0,
                    }
                ],
                "checkpoints": [
                    {
                        "checkpoint_steps": 0,
                        "metric_value": 0.5,
                        "meets_threshold": False,
                        "mean_attention_scale_grad_norm": None,
                        "attention_scale_stats": [initial],
                    },
                    {
                        "checkpoint_steps": 10,
                        "metric_value": 0.7,
                        "meets_threshold": False,
                        "mean_attention_scale_grad_norm": 0.1,
                        "attention_scale_stats": [final],
                    },
                ],
                "attention_scale_stats_final": [final],
                "recovered": False,
                "recovery_checkpoint": None,
            }
        ],
    }


def _write_artifact(tmp_path, data):
    output_root = tmp_path / "outputs"
    source = output_root / "raw" / "bounded.json"
    source.parent.mkdir(parents=True)
    source.write_text(json.dumps(data))
    retained = {"model": "example/model", "source": "raw/bounded.json"}
    return output_root, source, retained


def test_bounded_manifest_matches_all_nine_config_outputs():
    assert len(RETAINED_BOUNDED_ATTENTION_RECOVERY_RUNS) == 9
    assert {
        row["source"] for row in RETAINED_BOUNDED_ATTENTION_RECOVERY_RUNS
    } == EXPECTED_SOURCES


def test_build_row_accepts_completed_failed_recovery(tmp_path):
    output_root, _, retained = _write_artifact(tmp_path, _artifact())

    row = build_row(output_root, retained)

    assert row["recovered"] is False
    assert row["first_recovery_steps"] is None
    assert row["recovery_checkpoint"] is None
    assert row["max_observed_scale_to_alpha_ratio"] == 1.0
    assert row["min_observed_denominator_to_alpha_ratio"] == 1.0
    assert row["all_observed_constraints_satisfied"] is True


def test_build_row_requires_checkpoint_only_after_recovery(tmp_path):
    data = _artifact()
    result = data["results"][0]
    result["checkpoints"][-1]["metric_value"] = 0.805
    result["checkpoints"][-1]["meets_threshold"] = True
    result["recovered"] = True
    output_root, source, retained = _write_artifact(tmp_path, data)
    checkpoint = source.with_name(f"{source.stem}.fn24.recovery.pt")
    checkpoint.write_bytes(b"bounded checkpoint")
    result["recovery_checkpoint"] = str(checkpoint)
    source.write_text(json.dumps(data))

    row = build_row(output_root, retained)

    assert row["recovered"] is True
    assert row["first_recovery_steps"] == 10
    assert row["recovery_checkpoint"] == str(checkpoint)
    assert row["recovery_checkpoint_bytes"] == len(b"bounded checkpoint")


def test_build_row_rejects_unbounded_artifact(tmp_path):
    data = _artifact()
    data["enforce_alpha_denominator_bound"] = False
    output_root, _, retained = _write_artifact(tmp_path, data)

    with pytest.raises(ValueError, match="not a strict alpha-denominator-bound"):
        build_row(output_root, retained)


def test_build_row_rejects_scale_ratio_above_one_without_tolerance(tmp_path):
    data = _artifact()
    data["results"][0]["attention_scale_stats_final"][0][
        "scale_to_alpha_ratio"
    ] = math.nextafter(1.0, math.inf)
    output_root, _, retained = _write_artifact(tmp_path, data)

    with pytest.raises(ValueError, match="scale_to_alpha_ratio <= 1"):
        build_row(output_root, retained)


def test_build_row_rejects_denominator_below_floor_without_tolerance(tmp_path):
    data = _artifact()
    stat = data["results"][0]["checkpoints"][-1]["attention_scale_stats"][0]
    stat["score_denominator"] = math.nextafter(40.0, 0.0)
    output_root, _, retained = _write_artifact(tmp_path, data)

    with pytest.raises(ValueError, match="score_denominator >= score_denominator_floor"):
        build_row(output_root, retained)


def test_build_row_uses_calibrated_effective_alpha_for_floor(tmp_path):
    data = copy.deepcopy(_artifact())
    result = data["results"][0]
    result["effective_alpha_s"] = 3.0
    result["attention_scaling_layers"][0]["alpha_s"] = 3.0
    output_root, _, retained = _write_artifact(tmp_path, data)

    with pytest.raises(ValueError, match=r"effective_alpha_s\*\*2 \* alpha_w\*\*2"):
        build_row(output_root, retained)
