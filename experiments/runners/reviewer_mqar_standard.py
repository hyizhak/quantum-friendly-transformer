"""Train a matched ordinary Transformer baseline using the frozen MQAR runner."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from experiments.lib.mqar_standard import MQARStandardTransformer
from experiments.runners import reviewer_mqar as core

original_sources = core.sources


def standard_sources():
    return original_sources() + [Path(__file__), ROOT / "experiments/lib/mqar_standard.py",
        ROOT / "experiments/configs/reviewer/MQAR_STANDARD_BASELINE_PROTOCOL.txt"]


if __name__ == "__main__":
    core.MQARTransformer = MQARStandardTransformer
    core.sources = standard_sources
    core.main()
