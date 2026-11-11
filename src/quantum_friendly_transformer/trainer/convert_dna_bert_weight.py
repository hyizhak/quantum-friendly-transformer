"""Compatibility entrypoint for the repository's convert_dna_bert_weight recipe."""

from quantum_friendly_transformer.trainer.legacy_recipes import run_legacy_trainer_recipe


def main() -> None:
    run_legacy_trainer_recipe('convert_dna_bert_weight')


if __name__ == "__main__":
    main()
