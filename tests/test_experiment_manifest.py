"""Experiment provenance remains usable outside a Git checkout."""

import hashlib
import json
import shutil
import subprocess

import pytest

from experiments.lib import experiment


def init_repository(path):
    if shutil.which("git") is None:
        pytest.skip("Git is needed to exercise checkout metadata")
    path.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(
        ["git", "-C", str(path), "-c", "user.name=Test", "-c",
         "user.email=test@example.invalid", "-c", "commit.gpgsign=false",
         "commit", "--allow-empty", "-qm", "Test checkout"], check=True,
    )
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True,
    ).strip()


@pytest.mark.parametrize("worktree", [False, True])
def test_manifest_uses_source_checkout_from_another_directory(tmp_path, monkeypatch, worktree):
    source = tmp_path / "checkout"
    expected = init_repository(source)
    if worktree:
        worktree_path = tmp_path / "worktree"
        subprocess.run(
            ["git", "-C", str(source), "worktree", "add", "--detach", str(worktree_path)],
            check=True, capture_output=True,
        )
        source = worktree_path
        assert (source / ".git").is_file()
    monkeypatch.setattr(experiment, "ROOT", source)
    monkeypatch.chdir(tmp_path)
    assert experiment.manifest({}, [])['git_commit'] == expected


def test_manifest_does_not_attribute_source_zip_to_parent_repository(tmp_path, monkeypatch):
    parent = tmp_path / "unrelated"
    init_repository(parent)
    source = parent / "extracted_source"
    source.mkdir()
    monkeypatch.setattr(experiment, "ROOT", source)
    monkeypatch.chdir(source)
    assert experiment.manifest({}, [])['git_commit'] is None


@pytest.mark.parametrize("error", [FileNotFoundError("git"), subprocess.CalledProcessError(128, "git")])
def test_manifest_keeps_source_hashes_when_git_is_unavailable(tmp_path, monkeypatch, error):
    (tmp_path / ".git").mkdir()
    source = tmp_path / "experiment.py"
    source.write_text("print('experiment')\n")
    monkeypatch.setattr(experiment, "ROOT", tmp_path)

    def unavailable(*args, **kwargs):
        raise error

    monkeypatch.setattr(experiment.subprocess, "check_output", unavailable)
    result = experiment.manifest({"seed": 11}, [source], smoke=True)
    assert result["git_commit"] is None
    assert result["source_sha256"] == {str(source): hashlib.sha256(source.read_bytes()).hexdigest()}
    assert result["config"] == {"seed": 11}
    assert result["smoke"] is True
    assert json.loads(json.dumps(result))["git_commit"] is None
