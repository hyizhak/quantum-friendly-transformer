"""Installed research entrypoints; fast checks require no model downloads."""

from pathlib import Path
import sys

import pytest

from quantum_friendly_transformer.experiments.cli import available_configs, available_runners, main
from quantum_friendly_transformer.experiments.lib.checkpoints import recovery_checkpoint_path
from quantum_friendly_transformer.experiments.lib.commands import load_config, merged_run_configs
from quantum_friendly_transformer.experiments.runners.recovery.dispatcher import build_command


def test_packaged_configs_dispatch_without_checkout(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    configs = [name for name in available_configs() if name.startswith("recovery/")]
    assert configs
    for name in configs:
        config = load_config(Path(name))
        for run in merged_run_configs(config):
            command = build_command(run, sys.executable)
            assert command[1] == "-m"
            assert command[2].startswith("quantum_friendly_transformer.experiments.")
            assert not any("/home/" in item or "/data/" in item for item in command[1:])


def test_dispatcher_cli_reads_bundled_config(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    main(["recovery.dispatcher", "--config", "recovery/mistral7b_sst2_fn24_bounded_alpha_attention.yaml"])
    output = capsys.readouterr().out
    assert "quantum_friendly_transformer.experiments.runners.recovery.glue_sequence_classifier" in output
    assert "--enforce-alpha-denominator-bound" in output
    assert not list(tmp_path.iterdir())  # A dry run neither trains nor writes outputs.


def test_diagnostic_cli_help_has_no_source_dependency(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["diagnostics.forward_stats", "--help"])
    assert exc.value.code == 0
    assert "--seq-lens" in capsys.readouterr().out
    assert "diagnostics.token_norm_across_layers" in available_runners()


def test_checkpoint_paths_follow_relocated_artifacts(tmp_path):
    source = tmp_path / "raw/run.json"
    source.parent.mkdir()
    checkpoint = source.with_name("run.fn24.recovery.pt")
    checkpoint.write_bytes(b"checkpoint")
    assert recovery_checkpoint_path(source, "experiments/outputs/raw/run.fn24.recovery.pt") == checkpoint
    with pytest.raises(ValueError, match="matching sibling"):
        recovery_checkpoint_path(source, "experiments/outputs/raw/another.fn24.recovery.pt")
    checkpoint.unlink()
    with pytest.raises(FileNotFoundError, match="missing recovery checkpoint"):
        recovery_checkpoint_path(source, checkpoint.name)
