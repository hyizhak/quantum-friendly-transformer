#!/usr/bin/env python
from pathlib import Path as _Path
import sys as _sys

for _qft_parent in _Path(__file__).resolve().parents:
    if (_qft_parent / "src").exists() and (_qft_parent / "experiments").exists():
        _sys.path.insert(0, str(_qft_parent))
        _sys.path.insert(0, str(_qft_parent / "src"))
        break

"""Compatibility wrapper for experiments.runners.reports.aggregate_normalisation."""

from experiments.runners.reports.aggregate_normalisation import main


if __name__ == "__main__":
    main()
