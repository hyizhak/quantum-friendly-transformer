# NLP Revision Experiments

The historical NLP implementations ship in `quantum_friendly_transformer.experiments`.
Use `qft-experiment --list` and `qft-experiment --list-configs` after installation.
This directory also contains the later reviewer experiments, which run from a
source checkout using the commands below. Historical recorded results and
metadata ship with `quantum_friendly_transformer.evidence`; raw measurements
and checkpoints use its companion archive. See [the evidence guide](../docs/evidence.md)
for extraction and verification. The later reviewer results and checkpoints
are separate from that packaged snapshot.

This directory contains the runnable experiment handoff for the NLP revision work.
Historical scripts write raw outputs under `experiments/outputs/raw/` and metadata
beside each raw file as `*.metadata.json`. Reviewer runners use explicit output
directories, with local runs under `results/reviewer/` and independent
reproductions under `reproduced/`.

## Reviewer experiment map

All paths below are relative to `experiments/`. Run the entrypoints from the
repository root after installing `.[experiments,test]`.

| Experiment | Training entrypoint(s) in `runners/` | Protocol and configurations in `configs/reviewer/` |
|---|---|---|
| Paired recall through training length 1024 or 2048 | `reviewer_long_recall.py`, `reviewer_long_recall_routing.py` | [Long recall](configs/reviewer/LONG_RECALL_PROTOCOL.txt); `long_recall_{1024,2048}.json`, `long_recall_routing_{1024,2048}.json` |
| Native-temperature recall caps 24 and 64 | `reviewer_long_recall_routing.py`, `reevaluate_long_recall_caps.py` | [Native caps](configs/reviewer/NATIVE_RECALL_CAP_PROTOCOL.txt); `long_recall_native64_{1024,2048}.json` |
| Recall training through 8192 and validation-selected recovery | `reviewer_recall_8192.py`, `reviewer_recall_8192_recovery.py` | [Initial cohorts](configs/reviewer/RECALL_8192_CAP24_PROTOCOL.txt), [recovery search](configs/reviewer/RECALL_8192_RECOVERY_SEARCH.txt); `long_recall_train8192_*.json` |
| MQAR original and variance initialization | `reviewer_mqar.py`, `reviewer_mqar_variance_init.py` | [MQAR](configs/reviewer/MQAR_PROTOCOL.txt), [initialization follow-up](configs/reviewer/MQAR_INITIALIZATION_FOLLOWUP.txt); `mqar.json`, `mqar_variance_init.json` |
| MQAR ordinary Transformer baseline | `reviewer_mqar_standard.py` | [Standard baseline](configs/reviewer/MQAR_STANDARD_BASELINE_PROTOCOL.txt); `mqar_standard.json` |
| MQAR caps 48 and 24 | `reviewer_mqar_cap.py`, `reviewer_mqar_cap24.py` | [Cap48](configs/reviewer/MQAR_CAP_SENSITIVITY_PROTOCOL.txt), [cap24](configs/reviewer/MQAR_CAP24_PROTOCOL.txt); `mqar_cap48.json`, `mqar_cap24.json` |

The [long-recall reproduction guide](../docs/evidence.md#long-context-paired-recall-follow-up)
includes the classification-only and routing cohorts; the
[native-cap guide](../docs/evidence.md#comment-4-native-temperature-cap-comparison)
covers their matched cap comparison. Commands for MQAR and training through
8192 follow below. Audit and summary runners consume completed runs; use fresh
output directories for reproductions. Training sources, protocols, and pinned
vendor files have recorded hashes and must retain the bytes used by each run.

## MQAR benchmark

The MQAR reviewer benchmark uses the official Zoology generator pinned under
`vendor/zoology/`, with its original license and Git blob checksum. The protocol
is in [MQAR_PROTOCOL.txt](configs/reviewer/MQAR_PROTOCOL.txt). It evaluates a
two-layer capped Transformer and matched causal uniform-attention controls on
the standard 8192-token vocabulary, with no matching-index supervision. This is
a selected MQAR configuration, not the full published architecture sweep.

The original initialization cohort and the exploratory
[initialization follow-up](configs/reviewer/MQAR_INITIALIZATION_FOLLOWUP.txt)
are both retained. The follow-up changes initialization to accommodate unit-norm
activations; it keeps cap64, native attention temperature and the training budget.
The vocabulary readout is uncapped and separate from the capped layer matrices.

To reproduce both cohorts from the repository root, use fresh output directories:

```bash
python experiments/runners/reviewer_mqar.py prepare --output reproduced/mqar_data
python experiments/runners/run_mqar_cohort.py --variant native --device cuda:0 --assets reproduced/mqar_data --output reproduced/mqar
python experiments/runners/run_mqar_cohort.py --variant uniform --device cuda:1 --assets reproduced/mqar_data --output reproduced/mqar
python experiments/runners/reviewer_mqar_variance_init.py prepare --config experiments/configs/reviewer/mqar_variance_init.json --output reproduced/mqar_variance_init_data
python experiments/runners/run_mqar_cohort.py --variant native --device cuda:0 --runner experiments/runners/reviewer_mqar_variance_init.py --config experiments/configs/reviewer/mqar_variance_init.json --assets reproduced/mqar_variance_init_data --output reproduced/mqar_variance_init
python experiments/runners/run_mqar_cohort.py --variant uniform --device cuda:1 --runner experiments/runners/reviewer_mqar_variance_init.py --config experiments/configs/reviewer/mqar_variance_init.json --assets reproduced/mqar_variance_init_data --output reproduced/mqar_variance_init
python experiments/runners/audit_mqar.py --results reproduced/mqar --assets reproduced/mqar_data --device cuda:0
python experiments/runners/audit_mqar.py --results reproduced/mqar_variance_init --assets reproduced/mqar_variance_init_data --device cuda:0
python experiments/runners/summarize_mqar.py --initial-results reproduced/mqar --followup-results reproduced/mqar_variance_init --output reproduced/mqar_summary
```

Each variant coordinator retains all three fixed seeds and refuses to overwrite
an existing run. Different variants can run on separate GPUs. The audit checks
every saved query prediction and freshly replays the first batch of each test
condition for every checkpoint. The summary includes all seeds, both full
cohorts, a standalone PDF figure, and a LaTeX table. Tests are in `tests/test_mqar.py`.
These new records are local and are not included in the older reviewer release
manifest or any claimed Figshare deposit.

The matched ordinary Transformer baseline is defined separately in
`lib/mqar_standard.py` and
[MQAR_STANDARD_BASELINE_PROTOCOL.txt](configs/reviewer/MQAR_STANDARD_BASELINE_PROTOCOL.txt).
It uses ordinary trainable LayerNorm and uncapped projections, with the same
architecture dimensions, native attention denominator, data and training budget.
The earlier `native` run label denotes attention temperature within the capped
adaptation; it is not the ordinary Transformer baseline.

```bash
python experiments/runners/reviewer_mqar_standard.py prepare --config experiments/configs/reviewer/mqar_standard.json --output reproduced/mqar_standard_data
python experiments/runners/run_mqar_cohort.py --variant native --device cuda:0 --runner experiments/runners/reviewer_mqar_standard.py --config experiments/configs/reviewer/mqar_standard.json --assets reproduced/mqar_standard_data --output reproduced/mqar_standard
python experiments/runners/audit_mqar_standard.py --results reproduced/mqar_standard --assets reproduced/mqar_standard_data --reference-assets reproduced/mqar_variance_init_data --device cuda:0
```

For the recorded local cohorts, `python experiments/runners/summarize_mqar_comparison.py`
writes the unambiguously labeled three-model comparison under
`results/reviewer/mqar_comparison/`. Its data audit verifies exact tensor equality
with the capped model's evaluation data. The comparator was added after the
earlier experiment, so the combined comparison remains exploratory.

The cap-sensitivity follow-up lowers only the cap from 64 to 48, preserving raw
initialization, the data, temperature, normalization and training budget. Its
protocol is [MQAR_CAP_SENSITIVITY_PROTOCOL.txt](configs/reviewer/MQAR_CAP_SENSITIVITY_PROTOCOL.txt).
It retrains both attention variants under the smaller cap and reports actual
cap activity. Use fresh output directories for reproduction:

```bash
python experiments/runners/reviewer_mqar_cap.py prepare --config experiments/configs/reviewer/mqar_cap48.json --output reproduced/mqar_cap48_data
python experiments/runners/run_mqar_cohort.py --variant native --device cuda:0 --runner experiments/runners/reviewer_mqar_cap.py --config experiments/configs/reviewer/mqar_cap48.json --assets reproduced/mqar_cap48_data --output reproduced/mqar_cap48
python experiments/runners/run_mqar_cohort.py --variant uniform --device cuda:1 --runner experiments/runners/reviewer_mqar_cap.py --config experiments/configs/reviewer/mqar_cap48.json --assets reproduced/mqar_cap48_data --output reproduced/mqar_cap48
python experiments/runners/audit_mqar_caps.py --results reproduced/mqar_cap48 --assets reproduced/mqar_cap48_data --reference-results reproduced/mqar_variance_init --reference-assets reproduced/mqar_variance_init_data --device cuda:0
```

For the recorded local cohorts, `python experiments/runners/summarize_mqar_caps.py`
writes the cap comparison, all per-seed scores, cap-activity diagnostics and a
standalone figure under `results/reviewer/mqar_cap_comparison/`. Neither the
cap64 results nor the ordinary Transformer results are overwritten.

The subsequent cap24 cohort uses the same frozen training/model code and the
same data, raw initial parameters and training budget. Its protocol is
[MQAR_CAP24_PROTOCOL.txt](configs/reviewer/MQAR_CAP24_PROTOCOL.txt). It is a fresh
three-seed cohort with both trained attention and uniform controls:

```bash
python experiments/runners/reviewer_mqar_cap24.py prepare --config experiments/configs/reviewer/mqar_cap24.json --output reproduced/mqar_cap24_data
python experiments/runners/run_mqar_cohort.py --variant native --device cuda:0 --runner experiments/runners/reviewer_mqar_cap24.py --config experiments/configs/reviewer/mqar_cap24.json --assets reproduced/mqar_cap24_data --output reproduced/mqar_cap24
python experiments/runners/run_mqar_cohort.py --variant uniform --device cuda:1 --runner experiments/runners/reviewer_mqar_cap24.py --config experiments/configs/reviewer/mqar_cap24.json --assets reproduced/mqar_cap24_data --output reproduced/mqar_cap24
python experiments/runners/audit_mqar_caps.py --results reproduced/mqar_cap24 --assets reproduced/mqar_cap24_data --reference-results reproduced/mqar_variance_init --reference-assets reproduced/mqar_variance_init_data --device cuda:0
```

For the recorded local cohorts, `python experiments/runners/summarize_mqar_cap_series.py`
creates a separate comparison of caps 64, 48, 24 and the standard Transformer in
`results/reviewer/mqar_cap24_comparison/`, including every seed, corresponding
uniform controls, actual cap activity and figures. The earlier cap48 comparison
remains available with its original source hashes and reports.

## Cap24 training through length 8192

This follow-up uses the earlier continuous-key, balanced **single-query**
retrieval task, separately from MQAR. The initial
[8192 protocol](configs/reviewer/RECALL_8192_CAP24_PROTOCOL.txt) keeps the
architecture, native attention denominator, input norms and all six matrix
caps fixed, and changes the last curriculum stage to length8192. It retains
the disclosed matching-position auxiliary training loss. The initial cap24,
cap64 reference and uniform-cap24 cohorts each contain seeds11,22,33.

The subsequent [recovery search](configs/reviewer/RECALL_8192_RECOVERY_SEARCH.txt)
retains all four attempted training recipes: longer training from scratch and
three constrained fine-tuning recipes initialized from the cap64 checkpoints.
Loading those checkpoints into the cap24 model preserves maximum24 on every
projection; the cap64 maximum is not loaded from the parameter state.
Warm recipes receive the parent's3000 training updates as an extra resource.

The frozen evaluator uses filenames such as `test.csv` during training, but
these search outputs are explicitly marked **validation** in configurations
and manifests. Selection uses the mean8192 validation accuracy across all
three seeds. `select_recall_8192.py` freezes that choice before
`evaluate_recall_8192_final.py` opens the independent reserved test stream.

Example training commands, from the repository root into fresh directories:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 python experiments/runners/reviewer_recall_8192.py --config experiments/configs/reviewer/long_recall_train8192_cap24.json --output reproduced/recall8192_cap24 --variant native_sqrt_d --device cuda:0
CUBLAS_WORKSPACE_CONFIG=:4096:8 python experiments/runners/reviewer_recall_8192.py --config experiments/configs/reviewer/long_recall_train8192_cap24.json --output reproduced/recall8192_cap24 --variant uniform --device cuda:1
```

Use the cap64 configuration for the reference. The recovery runner is
`reviewer_recall_8192_recovery.py`; the four configurations are named
`long_recall_train8192_{cold_long,warm_route1,warm_route01,warm_answer}.json`.
For an independent reproduction, copy each warm configuration and set its
`warm_start_root` to the reproduced cap64 reference directory.
`audit_recall_8192.py` accepts the completed directories, checks stored
predictions and caps, and freshly replays the first batch at every length.
The recorded comparison is summarized under
`results/reviewer/long_recall_train8192_summary/` with the selection ledger,
all attempted recipes and reserved-test results. An unsuccessful recipe is
retained and does not justify a claim of recovery.

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
- `configs/reviewer/` contains the reviewer experiment protocols and fixed cohort configurations.
- Top-level `runners/reviewer_*.py`, `audit_*.py`, `validate_*.py`, and `summarize_*.py` implement reviewer training, auditing, and reporting from a source checkout.
- `runners/recovery/`, `runners/normalisation/`, `runners/diagnostics/`, and `runners/reports/` contain executable backends grouped by purpose.
- `runners/legacy/` contains the original fixed training and checkpoint-conversion recipes. The old package paths forward here from a source checkout; recipe bodies are not shipped in the library wheel.
- `scripts/` contains thin compatibility wrappers for the old flat paths. Prefer `runners/` paths or the config dispatchers for new commands. These wrappers can be deleted if no external workflow still calls the old `experiments/scripts/...` paths.
- `vendor/zoology/` retains the MQAR generator and license with their pinned upstream Git blob hashes.

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
