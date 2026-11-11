from pathlib import Path as _Path
import sys as _sys

for _qft_parent in _Path(__file__).resolve().parents:
    if (_qft_parent / "src").exists() and (_qft_parent / "experiments").exists():
        _sys.path.insert(0, str(_qft_parent))
        _sys.path.insert(0, str(_qft_parent / "src"))
        break

"""Compatibility imports for older experiment scripts."""

from experiments.lib.common import *  # noqa: F401,F403
