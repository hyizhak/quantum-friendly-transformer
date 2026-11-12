# Recorded experimental data

These files contain completed experimental results: per-seed metrics,
per-example predictions, training curves, matrix norms, and figure source data.
They are included in a GitHub source ZIP. No training or model download is
needed to inspect or aggregate them. Model weights, raw tensor datasets, and
local workflow-validation runs are excluded.

`results/inventory.json` lists each cohort archive and every member's SHA-256.
CSV prediction values are retained at their recorded precision. Metadata paths
use portable collection names and local cache fields are omitted. The inventory
records both the original and distributed hashes when metadata changes.
Checkpoint hashes and recorded training-source hashes identify historical
artifacts; they do not assert that training was rerun with the reorganized code.

| Data | Archives / tables |
|---|---|
| MQAR cap 24, trained averaging, and averaging substitution | `results/mqar_cap24.zip` |
| Ordinary Transformer reference | `results/mqar_standard.zip` |
| MQAR original initialization, adjusted initialization at cap 64, and cap 48 | `results/mqar.zip`, `results/mqar_variance_init.zip`, `results/mqar_cap48.zip` |
| MQAR primary, length/load, causal-prefix, and matrix-norm tables | `mqar/*.csv`, `mqar/summary.json` |
| Original SST-2 retained runs, seed 42, best and final validation checkpoints | `results/original_sst2.zip` |
| Later SST-2 recovery and matched controls | `results/recovery_*.zip` |
| Long-context recall, cap comparisons, validation recipe search, final evaluation | `results/long_recall_*.zip`, `results/native_cap_comparison.zip`, `results/recall.zip` |
| Pretrained and synthetic attention diagnostics | `results/pretrained.zip`, `results/synthetic.zip`, `results/selective_layer.zip` |

All completed seeds are retained, including poor results in the original MQAR
initialization cohort. Selection studies retain their validation records and
the separate final evaluation. The original SST-2 rows are single-seed
best-validation results, distinct from the later recovery studies.

From the repository root, verify all archived data with Python's standard library:

```bash
python tools/pack_result_data.py --verify artifacts/results
```

Recompute the MQAR tables directly from saved predictions:

```bash
python experiments/runners/summarize_mqar_predictions.py \
  --capped-results artifacts/results/mqar_cap24.zip \
  --standard-results artifacts/results/mqar_standard.zip \
  --output reproduced/mqar_tables
```

The command checks every saved prediction against its run's metrics, verifies
matched query identities across models/seeds, and computes the sample standard
deviation across seeds 11, 22, and 33. It also reports the subset whose causal
prefix length is at least the width (`position + 1 >= 64`). Training uses 20,000
sequences, 32 epochs, and 5,024 updates; each model is evaluated at its fixed
final checkpoint on 1,000 test sequences per condition.

At N=256 with 16 key-value pairs, the saved results are:

| Model | All 16,000 queries (%) | 8,168 queries with prefix length >= 64 (%) |
|---|---:|---:|
| Cap 24 | 98.671 ± 0.149 | 98.625 ± 0.138 |
| Causal average, trained | 0.294 ± 0.029 | 0.224 ± 0.058 |
| Causal average, substituted | 0.077 ± 0.036 | 0.033 ± 0.035 |
| Ordinary Transformer | 99.835 ± 0.052 | 99.792 ± 0.065 |

To use other reporting scripts, extract a cohort to a directory of your choice:

```bash
python -m zipfile -e artifacts/results/mqar_cap24.zip reproduced/mqar_cap24
```

Reaggregation checks stored records. Fresh inference requires the omitted
checkpoints or retraining with the [experiment configurations](../experiments/README.md).
The separate [archive workflow validator](../tools/validate_mqar_archive.py)
tests source-ZIP preparation, training, checkpoint loading, audits, and plots;
its default reduced training budget does not reproduce these numerical results.
