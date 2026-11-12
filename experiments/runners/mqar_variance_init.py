"""Exploratory MQAR initialization follow-up; retain the frozen original cohort."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from experiments.lib.mqar_variance_init import MQARVarianceInit
from experiments.runners import mqar as core

original_sources = core.sources


def followup_sources():
    return original_sources() + [Path(__file__), ROOT / "experiments/lib/mqar_variance_init.py",
        ROOT / "experiments/configs/mqar/MQAR_INITIALIZATION_FOLLOWUP.txt"]


if __name__ == "__main__":
    core.MQARTransformer = MQARVarianceInit
    core.sources = followup_sources
    core.main()
