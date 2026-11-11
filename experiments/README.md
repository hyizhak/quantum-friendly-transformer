# NLP Revision Experiments

The NLP implementations now ship in `quantum_friendly_transformer.experiments`.
Use `qft-experiment --list` and `qft-experiment --list-configs` after installation.
Paths under this directory remain compatibility entrypoints. Recorded results
and metadata ship with `quantum_friendly_transformer.evidence`; raw measurements
and checkpoints use its companion archive. See [the evidence guide](../docs/evidence.md)
for extraction and verification. The legacy training recipes remain local here.

This directory contains the runnable experiment handoff for the NLP revision work.
Experiment scripts write raw outputs under `experiments/outputs/raw/` and metadata
beside each raw file as `*.metadata.json`.

## Environment

Install the research dependencies from the repository root with
`python -m pip install -e '.[experiments]'`. Use `.[dev]` to include tests and
build tools. The core library can be installed separately with `pip install .`.

The repository-level `setup.sh` expects conda. On machines without conda, create
an equivalent Python environment and install the packages in
`experiments/environment.yml` manually. The scripts expect PyTorch,
Transformers, Datasets, pandas, pyarrow, scikit-learn, matplotlib, and PyYAML.
Use `huggingface-cli login` or set `HF_TOKEN` before commands that download gated
models.

## Layout

- `lib/` contains reusable experiment support: common I/O/metrics, command/config helpers, aggregation math, and retained-result manifests.
- `runners/recovery/`, `runners/normalisation/`, `runners/diagnostics/`, and `runners/reports/` contain executable backends grouped by purpose.
- `runners/legacy/` contains the original fixed training and checkpoint-conversion recipes. The old package paths forward here from a source checkout; recipe bodies are not shipped in the library wheel.
- `scripts/` contains thin compatibility wrappers for the old flat paths. Prefer `runners/` paths or the config dispatchers for new commands. These wrappers can be deleted if no external workflow still calls the old `experiments/scripts/...` paths.

## Library FN Caps

Learnable Frobenius caps are implemented in the package, not only in experiment
helpers. `frobenius_cap.py` is the sole definition of the learnable-cap adapter;
the Hugging Face runners import it:

```python
from quantum_friendly_transformer.norm_transformer import (
    apply_learnable_frobenius_caps,
    collect_learnable_cap_stats,
)

applied = apply_learnable_frobenius_caps(model, max_gamma=24.0)
cap_stats = collect_learnable_cap_stats(model)
```

This cap leaves a target weight unchanged when `||W||_F <= gamma` and scales it
only when `||W||_F > gamma`, with learned `gamma <= max_gamma`. It is different
from the older full Frobenius normalization APIs, which always normalize the
selected weight to a gamma-scaled Frobenius norm.

## Recovery Entrypoint

Use the recovery dispatcher as the public entrypoint for recovery runs. It reads YAML,
prints the resolved backend command by default, and executes only with
`--execute`.

```bash
python experiments/runners/recovery/dispatcher.py --config experiments/configs/recovery/tinystories_sst2_strict_baselines.yaml
python experiments/runners/recovery/dispatcher.py --config experiments/configs/recovery/tinystories_sst2_fn24_strict_recovery.yaml
python experiments/runners/recovery/dispatcher.py --config experiments/configs/recovery/olmo_1b_sst2_fn24.yaml
python experiments/runners/recovery/dispatcher.py --config experiments/configs/recovery/mistral7b_sst2_fncap.yaml
```

To run a config after checking the printed command:

```bash
python experiments/runners/recovery/dispatcher.py --config experiments/configs/recovery/mistral7b_sst2_fncap.yaml --execute
```

Use `--python /path/to/python` when the command should launch a specific
environment:

```bash
python experiments/runners/recovery/dispatcher.py \
  --config experiments/configs/recovery/tinystories_sst2_fn24_strict_recovery.yaml \
  --python /path/to/python \
  --execute
```

The config schema supports a single run:

```yaml
protocol: glue_sequence_classifier
args:
  model: mistralai/Mistral-7B-v0.1
  task: sst2
  caps: [24, 48, 96]
  checkpoints: [0, 100, 300, 500]
  output: experiments/outputs/raw/example.json
```

It also supports `defaults:` plus a `runs:` list for model ladders. The retained TinyStories ladder configs use this form.

Supported recovery protocols:

- `glue_sequence_classifier`: modern GLUE sequence-classifier recovery backend used for TinyStories, OLMo, and Mistral retained runs.
- `glue_encoder`: older encoder GLUE learnable-cap backend.
- `wikitext_lm`: WikiText causal-LM learnable-cap backend.
- `glue_warmstart`: warm-start GLUE cap recovery backend.
- `glue_distill`: warm-start plus teacher-logit distillation backend.
- `opt_wikitext_sweep`: OPT-family WikiText sweep backend.

## Retained Aggregates

Rebuild and validate the FN24 plus trainable-attention recovery matrix:

```bash
python experiments/runners/reports/aggregate_trainable_attention_recovery.py
```

This requires all nine retained JSON artifacts and recovery checkpoints. It
fails if a model is missing, did not recover, froze its scale controller, lacks
task-trainable scale evidence, or has no saved checkpoint. It writes CSV/JSON
under `experiments/outputs/aggregate_trainable_attention_recovery/` and the
human-readable summary at
`results/sst2_fn24_trainable_attention_recovery.md`.

Rebuild the filtered retained recovery table:

```bash
python experiments/runners/reports/aggregate_recovery_fn_vs_d.py
```

This writes
`experiments/outputs/aggregate_recovery_fn_vs_d/recovery_fn_vs_d_sst2.csv` for the
retained TinyStories SST-2 FN24 ladder plus the retained OLMo-1B/2048 and
Mistral-7B/4096 FN24 rows. The table reports baseline best/final metrics and
FN24 first-recovery, best, and final validation metrics.

Rebuild the retained decoder Z/N-vs-N aggregate tables from their raw parquet
files:

```bash
python experiments/runners/reports/aggregate_n_scaling_extended.py
```

The retained decoder set covers N=512,1024,2048,4096,8192 for
`unsloth/Llama-3.2-1B-Instruct`, `mistralai/Mistral-7B-v0.1`,
`Qwen/Qwen1.5-1.8B`, `Qwen/Qwen2.5-1.5B`, `Qwen/Qwen3-1.7B`,
`microsoft/phi-4`, `google/gemma-2b`, `google/gemma-3-4b-pt`, and
`bigscience/bloom-1b7`.

LayerNorm/RMSNorm denominator-vs-d retained summaries live in
`experiments/outputs/aggregate_dscale_layernorm/` with figures named
`experiments/outputs/figures/figure_layernorm_factor_vs_d_*`.

## Legacy Training Entrypoint

The older paper-reproduction trainers in `src/quantum_friendly_transformer/trainer/`
are import-safe compatibility wrappers. Use the experiment launcher when invoking
them from this tree:

```bash
python experiments/runners/training.py --list-recipes
python experiments/runners/training.py --config experiments/configs/training/legacy_trainers.yaml
```

Like the recovery dispatcher, `run_training.py` prints commands by default and executes
only with `--execute`.

## Forward Diagnostics

A minimal diagnostic run:

```bash
python experiments/runners/diagnostics/forward_stats.py --model FacebookAI/roberta-base --dataset glue --task mrpc --split validation --max-samples 1000 --seq-lens 128 256 512 --batch-size 16 --cache-dir /path/to/huggingface-cache --output experiments/outputs/raw/diag_roberta_mrpc.parquet
```

Measure per-token L2 norms across the residual stream of Llama 2 7B and
Qwen2.5 14B on eight matched, non-overlapping WikiText test chunks:

```bash
python experiments/runners/diagnostics/token_norm_across_layers.py
```

This writes token-level values to
`experiments/outputs/raw/token_l2_norm_across_layers.parquet`, layer summaries
and a constancy verdict (all block medians within 5% of the first block) to
`experiments/outputs/aggregate_token_norm_across_layers/`, and PNG/PDF figures
to `experiments/outputs/figures/figure_token_l2_norm_across_layers.*`.
The residual stream is measured after the embedding and every decoder block;
the model's final RMS-normalized output is shown separately. The same run also
records the output of every block's input RMSNorm—the exact token tensor sent
into self-attention—and writes
`figure_attention_input_l2_norm_across_layers.{png,pdf}` plus a separate
5%-tolerance constancy verdict. It additionally records the normalized vector
before the learned RMSNorm affine weights and compares its L2 norm with the
architectural invariant `sqrt(hidden_dim)`.

## Smoke Test

Validate hooks and aggregation without downloads or GPU work:

```bash
/path/to/python experiments/runners/diagnostics/smoke_local_models.py
/path/to/python experiments/runners/reports/aggregate_diagnostics.py --input-glob experiments/outputs/raw/smoke_local_diagnostics.parquet --output-dir /tmp/qft_smoke_aggregate
```

## Suite Runner

`run_experiment_suite.py` is kept for the older minimum/pilot diagnostic and
normalisation flows. New recovery runs should use
`experiments/runners/recovery/dispatcher.py` with a YAML config.

```bash
/path/to/python experiments/runners/suite.py --profile pilot
```
