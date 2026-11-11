"""Compatibility entrypoint for the repository's train_genomic_bench recipe."""

from quantum_friendly_transformer.trainer.legacy_recipes import run_legacy_trainer_recipe


def main() -> None:
    run_legacy_trainer_recipe('train_genomic_bench')


if __name__ == "__main__":
    main()
