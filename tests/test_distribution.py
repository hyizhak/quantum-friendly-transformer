"""Build and inspect both artifacts without downloads or model execution."""

import email
import io
import json
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path


def test_source_distribution_builds_a_self_contained_wheel(tmp_path):
    root = Path(__file__).resolve().parents[1]
    source = tmp_path / "source"
    source.mkdir()
    for name in ["pyproject.toml", "MANIFEST.in", "LICENSE", "README.md"]:
        shutil.copy2(root / name, source / name)
    (source / "response.tex").write_text("Private correspondence")
    for name in ["src", "docs", "examples", "tests", "tools"]:
        shutil.copytree(
            root / name,
            source / name,
            ignore=shutil.ignore_patterns("__pycache__", "*.egg-info", ".pytest_cache"),
        )
    # Accidental research outputs must not enter either release artifact.
    for directory in ["experiments", "alpha_s", "alpha_w", "results"]:
        output = source / directory
        output.mkdir()
        (output / "research-output.txt").write_text("excluded")
    dist = tmp_path / "dist"
    completed = subprocess.run(
        [sys.executable, "-m", "build", "--no-isolation", "--outdir", str(dist), str(source)],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    wheel, = dist.glob("*.whl")
    archive, = dist.glob("*.tar.gz")
    with tarfile.open(archive) as package:
        source_members = package.getnames()
    with zipfile.ZipFile(wheel) as package:
        members = package.namelist()
        metadata_file, = [name for name in members if name.endswith(".dist-info/METADATA")]
        metadata = email.message_from_bytes(package.read(metadata_file))
        assert metadata["Name"] == "quantum-friendly-transformer"
        assert metadata["Requires-Python"] == ">=3.10"
        requirements = metadata.get_all("Requires-Dist")
        assert [r for r in requirements if ";" not in r] == ["torch<3,>=2.4"]
        assert set(metadata.get_all("Provides-Extra")) == {"hf", "experiments", "test", "dev"}
        entry_points, = [name for name in members if name.endswith("entry_points.txt")]
        assert "qft-evidence" in package.read(entry_points).decode()
        assert "qft-experiment" in package.read(entry_points).decode()
        assert any(name.endswith("licenses/LICENSE") for name in members)
        assert "quantum_friendly_transformer/norm_transformer/bert/__init__.py" in members
        assert "quantum_friendly_transformer/models.py" in members
        assert "quantum_friendly_transformer/data.py" in members
        assert "quantum_friendly_transformer/experiments/runners/diagnostics/forward_stats.py" in members
        assert "quantum_friendly_transformer/experiments/runners/recovery/glue_sequence_classifier.py" in members
        assert "quantum_friendly_transformer/experiments/styles/qft-paper-v1.mplstyle" in members
        assert "quantum_friendly_transformer/evidence/catalog.json" in members
        assert "quantum_friendly_transformer/evidence/records.zip" in members
        catalog = json.loads(package.read("quantum_friendly_transformer/evidence/catalog.json"))
        assert all(row["path"] != "response.tex" for row in catalog["artifacts"])
        with zipfile.ZipFile(io.BytesIO(package.read("quantum_friendly_transformer/evidence/records.zip"))) as records:
            assert "response.tex" not in records.namelist()
        # Historical script modules must be shims, not the actual training recipes.
        shim = package.read("quantum_friendly_transformer/trainer/train_gue.py").decode()
        assert "run_legacy_trainer_recipe" in shim
        assert "load_dataset" not in shim
    for names in [members, source_members]:
        assert not any("research-output" in name for name in names)
        assert not any(name.endswith("/response.tex") for name in names)
        assert not any(name.endswith((".pt", ".pth", ".parquet", ".npy")) for name in names)
    assert any(name.endswith("/tests/test_library.py") for name in source_members)
    assert any(name.endswith("/tests/test_recovery_trainable_attention_scaling.py") for name in source_members)
    assert any(name.endswith("/tests/test_evidence.py") for name in source_members)
    assert not any(name.startswith("tests/") for name in members)

    # Import the built wheel from another directory, with no access to the
    # source package through the current process's editable installation.
    installed = tmp_path / "installed"
    with zipfile.ZipFile(wheel) as package:
        package.extractall(installed)
    script = f'''
import sys
sys.path.insert(0, {str(installed)!r})
from quantum_friendly_transformer.evidence import EvidenceStore
from quantum_friendly_transformer.experiments.cli import config_text, available_runners
from quantum_friendly_transformer.experiments.lib.commands import load_config
from pathlib import Path
import quantum_friendly_transformer as qft
assert Path(qft.__file__).is_relative_to(Path({str(installed)!r}))
store = EvidenceStore(cache_dir={str(tmp_path / 'cache')!r})
rows = store.read_csv("experiments/outputs/aggregate_bounded_attention_recovery/summary.csv")
assert len(rows) == 9
assert "diagnostics.forward_stats" in available_runners()
config = load_config(Path("recovery/mistral7b_sst2_fn24_bounded_alpha_attention.yaml"))
assert config["args"]["enforce_alpha_denominator_bound"] is True
assert "cache_dir" not in config["args"]
assert not any(name == "experiments" or name.startswith("experiments.") for name in sys.modules)
'''
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script], cwd=tmp_path, capture_output=True, text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
