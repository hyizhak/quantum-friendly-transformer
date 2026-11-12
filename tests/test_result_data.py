import importlib.util
import json
from pathlib import Path
import zipfile

import pytest

from experiments.runners.summarize_mqar_predictions import aggregate, score


def packer():
    path = Path(__file__).resolve().parents[1] / "tools" / "pack_result_data.py"
    spec = importlib.util.spec_from_file_location("pack_result_data", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_result_export_preserves_numbers_and_checks_distributed_bytes(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    record = {"evaluation": {"accuracy": 0.875, "correct": 7}, "cache_dir": str(tmp_path),
              "parent_checkpoint": {"path": str(source / "selected.pth"), "sha256": "original-weight-digest"}}
    (source / "result.json").write_text(json.dumps(record))
    (source / "predictions.csv").write_bytes(b"label,prediction\r\n1,1\r\n")
    (source / "selected.pth").write_bytes(b"weights")
    (source / "train.pt").write_bytes(b"dataset")
    (source / "notes.tex").write_text("Unrelated prose")
    output = tmp_path / "data"
    module = packer()
    module.pack([("experiment", source)], output, [(str(source), "experiment")])
    assert module.verify(output)["records"] == 2
    with zipfile.ZipFile(output / "experiment.zip") as archive:
        exported = json.loads(archive.read("result.json"))
        assert exported["evaluation"] == record["evaluation"]
        assert exported["parent_checkpoint"] == {"path": "experiment/selected.pth", "sha256": "original-weight-digest"}
        assert "cache_dir" not in exported
        assert archive.read("predictions.csv") == (source / "predictions.csv").read_bytes()
    with (output / "experiment.zip").open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ValueError, match="checksum mismatch"):
        module.verify(output)


def test_validation_runs_cannot_enter_result_data(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "result.json").write_text('{"smoke": true}')
    with pytest.raises(ValueError, match="Validation-only"):
        packer().pack([("experiment", source)], tmp_path / "data")


def test_prefix_subset_uses_causal_length_and_rejects_duplicate_queries():
    rows = [
        {"sequence": "0", "position": "62", "label": "5", "prediction": "5", "uniform_prediction": "0"},
        {"sequence": "0", "position": "63", "label": "6", "prediction": "0", "uniform_prediction": "6"},
        {"sequence": "1", "position": "64", "label": "7", "prediction": "7", "uniform_prediction": "7"},
    ]
    result = score(rows, 64)
    assert result["queries"] == 3 and result["correct"] == 2
    assert result["prefix_queries"] == 2 and result["prefix_correct"] == 1
    assert result["prefix_substituted_correct"] == 2
    assert aggregate([0.0, 0.5, 1.0]) == {"mean": 0.5, "sd": 0.5}
    with pytest.raises(ValueError, match="Duplicate query"):
        score(rows + rows[:1], 64)
