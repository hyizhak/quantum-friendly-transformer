"""Compatibility entrypoint for the repository's fine_tune_sn_conll2003 recipe."""

from quantum_friendly_transformer.trainer.legacy_recipes import run_legacy_trainer_recipe


def main() -> None:
    run_legacy_trainer_recipe('fine_tune_sn_conll2003')


if __name__ == "__main__":
    main()
