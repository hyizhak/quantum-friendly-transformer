"""MQAR cap24 cohort using the unchanged model and fixed-budget training code."""

from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/"src")]

from experiments.lib.mqar_variance_init import MQARVarianceInit
from experiments.runners import reviewer_mqar as core

original_sources=core.sources


def cap24_sources():
    return original_sources()+[Path(__file__),ROOT/"experiments/lib/mqar_variance_init.py",
        ROOT/"experiments/configs/reviewer/MQAR_CAP24_PROTOCOL.txt"]


if __name__=="__main__":
    core.MQARTransformer=MQARVarianceInit
    core.sources=cap24_sources
    core.main()
