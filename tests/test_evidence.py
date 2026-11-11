"""Evidence access needs no downloads, GPU, or real checkpoint deserialization."""

import hashlib
import json
from pathlib import Path
import zipfile

import pytest

from quantum_friendly_transformer.evidence import EvidenceStore
from quantum_friendly_transformer.evidence import store as store_module


@pytest.fixture
def fixture_store(tmp_path, monkeypatch):
    resource = tmp_path / "resource"
    resource.mkdir()
    records = {
        "experiments/outputs/raw/run.json": b'{"accuracy": 0.8, "recovered": false}',
        "experiments/outputs/table/summary.csv": b"model,value\nexample,1.5\n",
        "experiments/outputs/raw/run.fn24.recovery.pt": b"example checkpoint bytes",
    }
    entries = [
        {"path": name, "size": len(data), "sha256": hashlib.sha256(data).hexdigest(),
         "bundled": not name.endswith(".pt"),
         "groups": ["checkpoints"] if name.endswith(".pt") else ["sst2"]}
        for name, data in records.items()
    ]
    (resource / "catalog.json").write_text(json.dumps({"schema_version": 1, "artifacts": entries}))
    with zipfile.ZipFile(resource / "records.zip", "w") as archive:
        for name, data in records.items():
            if not name.endswith(".pt"):
                archive.writestr(name, data)
    bundle = tmp_path / "companion.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        name = "experiments/outputs/raw/run.fn24.recovery.pt"
        archive.writestr(name, records[name])
    monkeypatch.setattr(store_module, "files", lambda name: resource)
    monkeypatch.delenv("QFT_EVIDENCE_ROOT", raising=False)
    monkeypatch.delenv("QFT_EVIDENCE_BUNDLE", raising=False)
    store = EvidenceStore(cache_dir=tmp_path / "cache")
    return store, resource, bundle


def test_records_available_without_checkout(fixture_store, tmp_path, monkeypatch):
    store, _, _ = fixture_store
    monkeypatch.chdir(tmp_path)
    assert store.read_json("run.json") == {"accuracy": 0.8, "recovered": False}
    assert store.read_csv("summary.csv") == [{"model": "example", "value": "1.5"}]
    assert store.verify() == {}
    assert len(store.list_artifacts("checkpoints", bundled=False)) == 1
    with pytest.raises(ValueError, match="Unknown evidence group"):
        store.list_artifacts("unknown")


def test_bulk_requires_explicit_source_and_verifies_contents(fixture_store, tmp_path):
    store, _, bundle = fixture_store
    with pytest.raises(FileNotFoundError, match="companion evidence bundle"):
        store.path("run.fn24.recovery.pt")
    assert len(store.verify(include_bulk=True)) == 1
    store.bundle = bundle
    assert store.path("run.fn24.recovery.pt").read_bytes() == b"example checkpoint bytes"
    assert store.verify(include_bulk=True) == {}
    extracted = list(store.extract(tmp_path / "restored", include_bulk=True))
    assert len(extracted) == 3


def test_corrupt_source_does_not_create_artifact(fixture_store):
    store, _, bundle = fixture_store
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr("experiments/outputs/raw/run.fn24.recovery.pt", b"corrupt")
    store.bundle = bundle
    with pytest.raises(ValueError, match="checksum mismatch"):
        store.path("run.fn24.recovery.pt")
    assert not list(store.cache_dir.rglob("*.pt"))
    assert len(store.verify(include_bulk=True)) == 1


def test_existing_outputs_are_not_overwritten(fixture_store, tmp_path):
    store, _, _ = fixture_store
    destination = tmp_path / "restored"
    extracted = list(store.extract(destination))
    changed = extracted[0]
    changed.write_bytes(b"new experiment data")
    with pytest.raises(ValueError, match="checksum mismatch"):
        list(store.extract(destination))
    assert changed.read_bytes() == b"new experiment data"


def test_explicit_root_is_verified_and_can_be_relocated(fixture_store, tmp_path):
    store, _, bundle = fixture_store
    store.bundle = bundle
    root = tmp_path / "restored"
    list(store.extract(root, include_bulk=True))
    local = EvidenceStore(root=root, cache_dir=tmp_path / "unused-cache")
    path = local.path("run.json")
    assert path == root / "experiments/outputs/raw/run.json"
    assert not local.cache_dir.exists()
    path.write_text("changed")
    with pytest.raises(ValueError, match="checksum mismatch"):
        local.path("run.json")


def test_relative_bundle_path_survives_working_directory_change(fixture_store, tmp_path, monkeypatch):
    _, _, bundle = fixture_store
    monkeypatch.chdir(tmp_path)
    store = EvidenceStore(bundle=bundle.name, cache_dir="cache")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert store.path("run.fn24.recovery.pt").read_bytes() == b"example checkpoint bytes"


def test_symlink_cannot_escape_extraction_directory(fixture_store, tmp_path):
    store, _, _ = fixture_store
    destination = tmp_path / "restored"
    destination.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (destination / "experiments").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="escapes destination"):
        list(store.extract(destination))
    assert not list(outside.iterdir())


def test_shipped_catalog_covers_records_measurements_and_checkpoints(tmp_path):
    store = EvidenceStore(cache_dir=tmp_path)
    assert {"sst2", "partition", "layernorm", "depth", "checkpoints"} <= set(store.groups)
    assert store.verify() == {}
    checkpoints = store.list_artifacts("checkpoints")
    assert len(checkpoints) == 13
    assert all(not item.bundled and item.related for item in checkpoints)
    summary = store.read_json("experiments/outputs/aggregate_bounded_attention_recovery/summary.json")
    assert len(summary["results"]) == 9
    assert sum(row["recovered"] for row in summary["results"]) == 3
    depth = store.artifact("token_l2_norm_across_layers.parquet")
    assert not depth.bundled
    assert "depth" in depth.groups


@pytest.mark.parametrize("name", ["../escape", "/absolute", "a/../../escape", "a\\escape"])
def test_catalog_rejects_unsafe_paths(fixture_store, name):
    _, resource, _ = fixture_store
    path = resource / "catalog.json"
    data = json.loads(path.read_text())
    data["artifacts"][0]["path"] = name
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Invalid evidence path"):
        EvidenceStore()
