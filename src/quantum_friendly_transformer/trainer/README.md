# Trainer Package

`quantum_friendly_transformer.trainer` contains reusable training helpers and
compatibility entrypoints for the original experiments. Recipe bodies live in
`experiments/runners/legacy/` and are excluded from the library distribution.

## Reusable API

- `util.py` exposes `train(...)` and `evaluate(...)` for the custom
  transformer models in `norm_transformer/`.
- The package exports these helpers as:

```python
from quantum_friendly_transformer.trainer import train, evaluate
```

## Legacy Recipes

The following filenames forward to fixed experiment recipes in a source checkout:

- `train_gue.py`, `train_genomic_bench.py`, and `train_conll2003.py` train
  vanilla, spectral-normalized, and Frobenius-normalized custom models.
- `fine_tune_sn_model_*.py` run spectral-normalized fine-tuning recipes.
- `fine_tune_multilayer_*.py` run older multi-layer BERT normalization recipes.
- `convert_dna_bert_weight.py` is a one-off checkpoint conversion utility.

These scripts intentionally stay as compatibility entrypoints. An installed wheel
can use `train` and `evaluate` but needs a source checkout to execute a recipe.
New experiment
orchestration should live under `experiments/` as config-driven runners that
call reusable package functions.


## Config-Driven Legacy Launcher

Legacy trainer recipes are registered in `legacy_recipes.py` and can be launched
from the experiment layer:

```bash
python experiments/runners/training.py --list-recipes
python experiments/runners/training.py --config experiments/configs/training/legacy_trainers.yaml
```

The launcher prints commands by default. Add `--execute` only after checking the
resolved commands and local output paths. Existing trainer files remain directly
executable from an installed source checkout; importing them does not load
optional training frameworks or start downloads or training.

## Refactor Rule

When a legacy trainer needs shared behavior, extract the reusable piece into the
package first, then make the legacy script call it. Do not add new hardcoded
dataset/model/output recipes to this package.
