"""Build and inspect both artifacts without downloads or model execution."""

import email
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
    for name in ["src", "docs", "examples", "tests"]:
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
        assert not any(name.endswith("entry_points.txt") for name in members)
        assert any(name.endswith("licenses/LICENSE") for name in members)
        assert "quantum_friendly_transformer/norm_transformer/bert/__init__.py" in members
        assert "quantum_friendly_transformer/models.py" in members
        assert "quantum_friendly_transformer/data.py" in members
        # Historical script modules must be shims, not the actual training recipes.
        shim = package.read("quantum_friendly_transformer/trainer/train_gue.py").decode()
        assert "run_legacy_trainer_recipe" in shim
        assert "load_dataset" not in shim
    for names in [members, source_members]:
        assert not any("research-output" in name for name in names)
        assert not any(name.endswith((".pt", ".pth", ".parquet", ".npy")) for name in names)
    assert any(name.endswith("/tests/test_library.py") for name in source_members)
    assert not any(name.endswith("/tests/test_recovery_trainable_attention_scaling.py") for name in source_members)
    assert not any(name.startswith("tests/") for name in members)
