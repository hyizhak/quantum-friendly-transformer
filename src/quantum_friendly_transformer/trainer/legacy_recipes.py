"""Compatibility launcher for recipes kept in a source checkout.

The distribution contains reusable trainers, not fixed datasets, local output
paths, or paper-reproduction recipes. These helpers locate the latter only
when running from the repository (including an editable installation).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LegacyTrainerRecipe:
    name: str
    script: str
    description: str


LEGACY_TRAINER_RECIPES = {
    "train_gue": LegacyTrainerRecipe(
        name="train_gue",
        script="train_gue.py",
        description="One-layer GUE promoter classification recipe.",
    ),
    "train_genomic_bench": LegacyTrainerRecipe(
        name="train_genomic_bench",
        script="train_genomic_bench.py",
        description="One-layer Genomic Benchmarks promoter classification recipe.",
    ),
    "train_conll2003": LegacyTrainerRecipe(
        name="train_conll2003",
        script="train_conll2003.py",
        description="One-layer CoNLL-2003 token classification recipe.",
    ),
    "fine_tune_sn_gue": LegacyTrainerRecipe(
        name="fine_tune_sn_gue",
        script="fine_tune_sn_model_gue.py",
        description="Spectral-normalized GUE fine-tuning recipe.",
    ),
    "fine_tune_sn_genomic_bench": LegacyTrainerRecipe(
        name="fine_tune_sn_genomic_bench",
        script="fine_tune_sn_model_genomic_bench.py",
        description="Spectral-normalized Genomic Benchmarks fine-tuning recipe.",
    ),
    "fine_tune_sn_conll2003": LegacyTrainerRecipe(
        name="fine_tune_sn_conll2003",
        script="fine_tune_sn_model_conll2003.py",
        description="Spectral-normalized CoNLL-2003 fine-tuning recipe.",
    ),
    "fine_tune_multilayer_gue": LegacyTrainerRecipe(
        name="fine_tune_multilayer_gue",
        script="fine_tune_multilayer_gue.py",
        description="Multi-layer Frobenius-normalized DNABERT GUE recipe.",
    ),
    "fine_tune_multilayer_genomic_bench": LegacyTrainerRecipe(
        name="fine_tune_multilayer_genomic_bench",
        script="fine_tune_multilayer_genomic_bench.py",
        description="Multi-layer Frobenius-normalized DNABERT Genomic Benchmarks recipe.",
    ),
    "convert_dna_bert_weight": LegacyTrainerRecipe(
        name="convert_dna_bert_weight",
        script="convert_dna_bert_weight.py",
        description="DNABERT checkpoint conversion utility for normalized BERT modules.",
    ),
}


def trainer_recipe_names() -> list[str]:
    return sorted(LEGACY_TRAINER_RECIPES)


def get_legacy_trainer_recipe(name: str) -> LegacyTrainerRecipe:
    try:
        return LEGACY_TRAINER_RECIPES[name]
    except KeyError as exc:
        allowed = ", ".join(trainer_recipe_names())
        raise ValueError(f"unknown legacy trainer recipe {name!r}; expected one of: {allowed}") from exc


def legacy_trainer_script_path(name: str) -> Path:
    recipe = get_legacy_trainer_recipe(name)
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            script = parent / "experiments" / "runners" / "legacy" / recipe.script
            if script.is_file():
                return script
    raise FileNotFoundError(
        f"Recipe {name!r} requires a source checkout. Clone "
        "https://github.com/hyizhak/quantum-friendly-transformer, install it "
        "with pip install -e '.[experiments]', and run "
        "experiments/runners/training.py from that checkout."
    )


def legacy_trainer_command(name: str, python: str, extra_args: list[str] | None = None) -> list[str]:
    command = [python, str(legacy_trainer_script_path(name))]
    if extra_args:
        command.extend(extra_args)
    return command


def run_legacy_trainer_recipe(name: str) -> None:
    """Execute a repository recipe through an old module/script entrypoint."""
    import runpy

    runpy.run_path(str(legacy_trainer_script_path(name)), run_name="__main__")
