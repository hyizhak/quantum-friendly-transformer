import json
from pathlib import Path
import sys

from experiments.runners import prepare_recovery_search_assets as prepare


def test_asset_preparation_downloads_without_a_machine_cache(tmp_path, monkeypatch):
    data = tmp_path / "dataset"
    data.mkdir()
    (data / "train.parquet").write_bytes(b"dataset fixture")
    recipe = tmp_path / "recipe.json"
    recipe.write_text(json.dumps({"models": {"1m": {"model_id": "example/tiny", "revision": "pinned"}},
                                  "dataset_id": "example/data", "dataset_revision": "data-pinned"}))
    output = tmp_path / "assets"
    downloads = []

    def download(model_id, *, revision, allow_patterns, local_dir, **kwargs):
        assert model_id == "example/tiny" and revision == "pinned"
        assert "pytorch_model*.bin" in allow_patterns
        local_dir.mkdir(parents=True)
        (local_dir / "pytorch_model.bin").write_bytes(b"weight fixture")
        downloads.append(local_dir)

    monkeypatch.setattr(prepare, "snapshot_download", download)
    monkeypatch.setattr(sys, "argv", ["prepare", "--config", str(recipe), "--data", str(data), "--output", str(output)])
    prepare.main()
    assert downloads == [output / "1m"]
    manifest = json.loads((output / "assets.json").read_text())
    assert set(manifest["sha256"]) == {"1m/pytorch_model.bin", "data/train.parquet"}
    assert (output / "data").resolve() == data


def test_asset_preparation_reuses_only_explicit_inputs(tmp_path, monkeypatch):
    shared = tmp_path / "existing"
    (shared / "1m").mkdir(parents=True)
    (shared / "1m/model.safetensors").write_bytes(b"weight fixture")
    data = tmp_path / "dataset"
    data.mkdir()
    (data / "train.parquet").write_bytes(b"dataset fixture")
    recipe = tmp_path / "recipe.json"
    recipe.write_text(json.dumps({"models": {"1m": {"model_id": "example/tiny", "revision": "pinned"}},
                                  "dataset_id": "example/data", "dataset_revision": "data-pinned"}))
    output = tmp_path / "assets"
    monkeypatch.chdir(tmp_path)

    def unexpected_download(*args, **kwargs):
        raise AssertionError("Explicitly supplied weights should be reused")

    monkeypatch.setattr(prepare, "snapshot_download", unexpected_download)
    monkeypatch.setattr(sys, "argv", ["prepare", "--config", str(recipe), "--data", "dataset",
                                     "--reuse-assets", "existing", "--output", str(output)])
    prepare.main()
    assert (output / "1m").resolve() == shared / "1m"
    manifest = json.loads((output / "assets.json").read_text())
    assert set(manifest["sha256"]) == {"1m/model.safetensors", "data/train.parquet"}
