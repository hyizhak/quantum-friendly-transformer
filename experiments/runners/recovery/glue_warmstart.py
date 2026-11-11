"""Compatibility entrypoint for :mod:`quantum_friendly_transformer.experiments.runners.recovery.glue_warmstart`."""

from pathlib import Path
import sys

for parent in Path(__file__).resolve().parents:
    if (parent / "src" / "quantum_friendly_transformer").is_dir():
        sys.path.insert(0, str(parent / "src"))
        break

if __name__ == "__main__":
    import runpy

    runpy.run_module("quantum_friendly_transformer.experiments.runners.recovery.glue_warmstart", run_name="__main__")
else:
    from importlib import import_module

    sys.modules[__name__] = import_module("quantum_friendly_transformer.experiments.runners.recovery.glue_warmstart")
