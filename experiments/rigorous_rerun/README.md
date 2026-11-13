# Classification reruns and result records

The delivered experiments use **seed 42 on human non-TATA promoters**. All
three single-layer runs and the multi-layer follow-ups have completed training
and evaluation. The multi-layer follow-up uses ordinary trainable LayerNorm with complete
projection Frobenius norms bounded by one. Metrics, predictions, histories and
run metadata remain in the local runtime directories. These follow-ups are
exploratory because benchmark test scores were available during development.

The original 105-run plan below is archived and stopped; it is not a completed
study and does not resume automatically. Selected checkpoint hashes were locked
before each corresponding test evaluation.

## Code and saved runs

| Entry point | Purpose |
|---|---|
| `prepare.py`, `run.py` | Audit splits, freeze source, train and evaluate the base study |
| `single_seed_delivery.py`, `paper_seed42_delivery.py` | Export the completed seed-42 base results |
| `unit_fn_search.py`, `test_unit_fn_variants.py` | Fixed-unit-LayerNorm FN1 conditions and their test evaluations |
| `recover_unit_fn_ffn.py`, `unit_fn_layerwise_distill.py` | Archived FFN and representation-training follow-ups |
| `test_unit_fn_layerwise.py` | Evaluate the two representation-training conditions |
| `trainable_ln_fn1.py` | Ordinary trainable LayerNorm with FN1; training, checkpoint locking and evaluation |

Follow-up runners load frozen dependencies from the recorded runtime directories
under `/data/hyizhan`. They require the original prepared tensors and checkpoints;
those raw run artifacts are not included in this source release.
`plan.json` records source/data hashes, objectives and optimizer settings.
Frozen scripts enforce these hashes, so use their recorded snapshots to resume
an existing run. Source changes belong to a new run with a new manifest.
Public exporters write numerical records only. Manuscript assembly code and
editorial verification records remain local, including original script snapshots
that contain manuscript text. Historical source hashes identify the external
frozen runners, not the public exporters after this cleanup.
The `test_unit_*.py` files are experiment evaluation commands, not pytest tests.

For code regression tests, run from the repository root:

```bash
python -m pytest -q
```

## Archived base-study protocol

The November 2026 rerun uses the corrected attention, RoPE, normalization and
checkpoint-loading implementations. Its protocol is
[`../configs/training/rigorous_rerun.json`](../configs/training/rigorous_rerun.json).
It contains **105 training runs**, using seeds 42, 3407, 2023, 2024 and 2025:

| Data | Single layer | SN fine-tuning | Pretrained multilayer |
|---|---|---|---|
| Genomic Benchmarks non-TATA | vanilla, SN, FN | attention only, FFN only | vanilla, FN, subnorm |
| GUE `prom_300_notata` | vanilla, SN, FN | attention only, FFN only | vanilla, FN, subnorm |
| CoNLL-2003 **POS tagging** | vanilla, SN, FN | attention only, FFN only | — |

Each variant uses all five seeds. No seed is chosen for favorable test results.
The single-layer and fine-tuning models have trainable pretrained token
embeddings; they are not full pretrained transformers. Multilayer models use
the pinned DNABERT-2 backbone. Its pooler and classifier are newly initialized,
then shared across the variants within each seed.

## Data and comparability

The prepared data and all checkpoints are on
`/data/hyizhan/qft-rigorous-rerun-20261112`, outside the nearly full root volume.
The pinned source revisions, source file checksums, tokenization and row
identities are recorded in `data/*/manifest.json` and `*_rows.json`.

| Data | Training rows | Validation rows | Test rows | Overlapping rows removed from train/dev |
|---|---:|---:|---:|---:|
| Genomic | 24,387 | 2,710 | 9,034 | 0 |
| GUE | 42,421 | 5,304 | 5,307 | 34 |
| CoNLL | 13,492 | 3,187 | 3,453 | 612 |

Duplicate groups join inputs with the same full text or effective token IDs.
For DNA, the full-sequence key also matches reverse complements. Test rows are
all retained; matching rows are removed from lower-priority splits. Repeated
inputs within a split remain, with group IDs for auditing. This does not prove
independence with respect to biological homology, genomic proximity or source
documents.

Genomic validation is a fixed, stratified 10% split of training groups. The old
code split the official test set into validation and test halves. The corrected
study therefore has a different evaluation protocol: changes from the old
table cannot be attributed solely to the bug fixes.

Other declared changes are paired raw initial weights, paired epoch sample
order, BF16 training with FP32 master parameters, deterministic CUDA execution,
FP32 evaluation, and validation-selected parents for SN fine-tuning instead of
fixed historical epochs. The epoch budgets, batch sizes, learning rates,
schedulers, weight decay and early-stopping settings are recorded in the
protocol. No budget is shortened for turnaround time.

The base-study multilayer variants use fixed unit LayerNorm, FFN residual
scaling and changed attention temperature. This differs from the older
trainable-LayerNorm model; the later `trainable_ln_fn1.py` cohort restores
ordinary trainable LayerNorm. These comparisons do **not** isolate the effect
of a norm constraint or establish accuracy with the theorem's attention scaling.
PyTorch SN uses approximate power iteration; final effective spectral norms
are independently measured with SVD rather than assumed to satisfy an exact cap.
Single-layer SN/FN constrain Q/K/V and the two FFN weight matrices; embeddings,
the attention output projection and the classifier remain unconstrained.

## Execution and outputs

`prepare.py` reads pinned local sources, audits duplicates and saves fixed
tensor datasets. `run.py freeze` copies the actual library and runner into
`source/`, records their SHA-256 hashes, validates the full-batch preflight,
and writes `study.json`. Workers execute that frozen copy and check the
source, model, tokenizer and dataset hashes before training.

`launch.py` supervises two GPU workers. Each dataset/seed group stays on one
GPU, so all its model variants use the same hardware. The two available GPUs
are A100-SXM4-40GB and A100-PCIE-40GB; cross-GPU bitwise equality is not claimed.
The supervisor holds a file lock to prevent duplicate launches. A numerical
failure stops the affected worker and is recorded; settings are not silently
changed or failing seeds discarded.

Each epoch writes resumable optimizer, scheduler and random-number state
atomically. Validation selects the first epoch achieving the highest metric;
epoch zero is diagnostic only. Every selected checkpoint is reloaded and its
validation scores checked before selection is locked. Test evaluation starts
only after **all 105 selections** are complete. The supervisor then writes
per-example test logits, predictions and labels, measured normalized-weight
norms, `summary.json`, and `summary.md`. Intervals describe variability across
five paired training seeds on the fixed split; they do not measure uncertainty
over data splits or populations.

To inspect progress without touching training:

```bash
PYTHONPATH=.:src .venv/bin/python -m experiments.rigorous_rerun.run status \
  --root /data/hyizhan/qft-rigorous-rerun-20261112
```

The runtime directory contains `supervisor_status.json`, `logs/queue_worker*.log`,
`runs/<dataset>/<seed>/<family>/<variant>/history.json`, and any failure records.
After diagnosing a stopped worker, the supervisor can resume from each last
completed epoch; do not launch a second supervisor while its lock is held.

The first startup was archived as `qft-rigorous-rerun-20261112-startup-io-v1`
after a concurrent JSON temporary-file collision was found in the final
evaluation orchestration. No test evaluation had occurred. The final study
restarted all runs from their fixed initial weights with unchanged data and
training settings; its amendment record links both source snapshots. The
overlapping actual training histories were checked for identical losses,
validation metrics, gradient norms, learning rates and sample order, excluding
elapsed time. Two independently reloaded saved checkpoints also reproduced
their recorded validation metrics exactly. The records remain in the local `artifacts/rigorous_rerun/` directory.

The SST-2 result archives contain historical JSON records, not model weights.
These legacy reruns do not regenerate the SST-2 table or establish that its
historical checkpoints were re-evaluated.
