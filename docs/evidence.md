# Experiments and recorded evidence

For the later reviewer experiments, see the
[experiment map](../experiments/README.md#reviewer-experiment-map). Their
`results/reviewer/` records and checkpoints are separate from the historical
snapshot below. Peer-review correspondence is excluded from the code and
bundled evidence. Local packaging does not publish an update to Figshare.

Version 0.3.0 includes the NLP research runners in
`quantum_friendly_transformer.experiments` and recorded evidence in
`quantum_friendly_transformer.evidence`. Both work from an installed wheel.
Installing the library does not run experiments or download model weights.

## What is distributed

| Content | Location |
|---|---|
| Recovery, forward diagnostics, normalization, depth measurements, reporting | Packaged Python modules under `quantum_friendly_transformer.experiments` |
| Portable YAML configs and plotting style | Package resources, available through the experiment CLI |
| SST-2 run JSON, measurement metadata, aggregate CSVs, result reports | `evidence/records.zip`, included in the wheel and source distribution |
| All recorded file paths, sizes, SHA-256 checksums, groups, related files | `evidence/catalog.json`, included in both distributions |
| Raw Parquet/NumPy measurements, the large token-level LayerNorm CSV, figures, saved trainable states | Companion ZIP, distributed separately from the Python package |

The current snapshot contains 1,305 files: 540 compact files in the library and
765 files in the companion bundle. There are 65 SST-2 JSON files excluding
metadata, 134 Parquet files overall, and 13 recovery checkpoint files.
The companion ZIP is about 5.84 GB; its filename for this version is
`quantum_friendly_transformer-0.3.0-evidence.zip`.

The library's compact archive is tracked as a package resource, so a fresh clone
and a built wheel contain actual records and tables. The companion archive is
created under `dist/` and must be shared separately. There is no automatic
remote download or assumed public archive URL. `qft-evidence list` clearly marks
each entry as `wheel` or `companion`.

All retained experiment file contents are preserved, including run failures, provenance,
sampling information, and historical paths. Catalog groups are discovery labels,
not validation that a run meets a paper claim. Some diagnostics omit entropy or
other measurements; inspect the saved metadata and columns before combining runs.
The retained bounded-attention report records three recoveries out of nine;
the separate trainable-attention report records nine out of nine. These use
different constraints. Packaging does not add seeds, held-out evaluations, or new
measurements.

## Read records through Python

The evidence API uses the standard library for JSON, CSV, cataloguing, extraction,
and integrity checks. The package retains its core PyTorch dependency, and
checkpoint loading uses PyTorch. pandas and pyarrow are optional readers for raw
tables.

```python
from quantum_friendly_transformer.evidence import EvidenceStore

evidence = EvidenceStore()
rows = evidence.read_csv(
    "experiments/outputs/aggregate_bounded_attention_recovery/summary.csv"
)
run = evidence.read_json(
    "recovery_sst2_tinystories_1m_fn24_bounded_alpha_attention_steps20000.json"
)
checkpoints = evidence.list_artifacts(group="checkpoints")
```

Use a full catalog path when filenames are ambiguous. `path()` returns a verified
local file, extracting it to the cache only when necessary. The default cache is
`$XDG_CACHE_HOME/qft/evidence` or `~/.cache/qft/evidence`; override it with
`cache_dir=` or `QFT_EVIDENCE_CACHE`. Importing the library creates no cache files.

To read raw measurements from the companion archive:

```python
import pandas as pd
from quantum_friendly_transformer.evidence import EvidenceStore

evidence = EvidenceStore(bundle="/path/to/quantum_friendly_transformer-0.3.0-evidence.zip")
depth = pd.read_parquet(evidence.path("token_l2_norm_across_layers.parquet"))
```

Alternatively use `EvidenceStore(root="/path/to/original-checkout")`. The root
must contain `experiments/outputs/` with the original relative paths. The
environment equivalents are `QFT_EVIDENCE_BUNDLE` and `QFT_EVIDENCE_ROOT`.

## Restore or verify the evidence

```bash
qft-evidence list --group sst2
qft-evidence list --group depth
qft-evidence verify
qft-evidence extract --destination ./research
```

By default, extraction and verification cover only bundled records. To include
the companion data:

```bash
qft-evidence verify --include-bulk --bundle /path/to/quantum_friendly_transformer-0.3.0-evidence.zip
qft-evidence extract --include-bulk --bundle /path/to/quantum_friendly_transformer-0.3.0-evidence.zip --destination ./research
```

The commands restore `experiments/outputs/` and `results/`
under the destination. Existing files with different contents cause an error;
use a new destination to preserve subsequent experiments. Checksums cover
uncompressed original bytes, whether the source is a directory or a ZIP.

## Run experiments and regenerate reports

Install the research dependencies with `pip install '.[experiments]'` from a
checkout, or `pip install '/path/to/library.whl[experiments]'` for a built wheel.
List the installed commands and configurations:

```bash
qft-experiment --list
qft-experiment --list-configs
qft-experiment diagnostics.forward_stats --help
qft-experiment diagnostics.token_norm_across_layers --help
qft-experiment recovery.dispatcher --config recovery/tinystories_sst2_fn24_bounded_alpha_attention.yaml
```

The dispatcher prints resolved commands by default; add `--execute` to train.
Named configs are read from package resources if no local file matches. Packaged
configs use the normal Hugging Face cache instead of this checkout's machine
paths. Historical configs retain `offline: true`; use `--set args.offline=false`
for a single-run config, or `--set defaults.args.offline=false` for a run list,
when model downloads are needed. Authentication, datasets, and suitable hardware
are required for those runs. Training hyperparameters and recorded results were
not changed during this reorganization.

These module invocations are equivalent to the console commands:

```bash
python -m quantum_friendly_transformer.evidence list
python -m quantum_friendly_transformer.experiments diagnostics.forward_stats --help
```

After extracting all evidence, rebuild the retained recovery reports from the
restored directory:

```bash
cd research
qft-experiment reports.aggregate_trainable_attention_recovery
qft-experiment reports.aggregate_bounded_attention_recovery
qft-experiment reports.aggregate_n_scaling_extended
```

Recovery validators require the matching checkpoint next to each successful
run's JSON. They resolve relocated sibling files rather than the machine paths
recorded during training. Measurement/report outputs default to the current
working directory's `experiments/outputs/`, never the installed package directory.
Regenerated reports may differ in path labels after relocation; use another
output directory if retaining the snapshot's checksums matters.

A complete CPU smoke workflow creates random tiny BERT/GPT-2/Llama models,
collects attention and norm diagnostics, aggregates them, and plots the results:

```bash
qft-experiment suite --profile smoke --output-root ./smoke-results --execute
```

It needs the `experiments` extra but no GPU or downloaded models. Original
`experiments/runners/` and `experiments/scripts/` paths remain compatibility
entrypoints in a checkout. The older genomic/CoNLL/GUE fixed recipes remain
under `experiments/runners/legacy/` and are described in the library guide.

## Saved checkpoints

```python
checkpoint = evidence.load_checkpoint(
    "recovery_sst2_mistral7b_fn24_trainable_alpha_attention_tasktrainable_steps500.fn24.recovery.pt"
)
print(checkpoint["format"], checkpoint["reconstruction_order"])
state = checkpoint["trainable_state_dict"]
```

Checkpoint loading verifies the catalog checksum and uses `torch.load` with
`weights_only=True` and CPU placement by default. These files store trainable
state, not uniformly complete pretrained models. Reconstruct the recorded base
model, optional PEFT adapter, Frobenius caps, and attention controller in the
saved order before loading that state. Frozen base parameters, tokenizer files,
and optimizer/resume state are not supplied by these checkpoint files.

## Rebuild a release snapshot

With the original experiment outputs available:

```bash
python tools/build_evidence.py --bundle dist/quantum_friendly_transformer-0.3.0-evidence.zip
python -m build
python -m twine check dist/*.whl dist/*.tar.gz
```

The builder creates deterministic ZIP entries, preserves all selected source
bytes, refreshes the catalog and compact archive, and writes a SHA-256 sidecar
for the companion ZIP. It refuses to overwrite an existing companion archive;
choose a new filename when creating a different snapshot. Standard wheel/sdist
builds use the checked-in catalog and compact archive and do not need the large
original files. Building a release does not publish it.

Fast tests are `tests/test_evidence.py`, `tests/test_experiments.py`, and the
recovery/library tests. `tests/test_distribution.py` builds the sdist and wheel,
then reads bundled evidence and configs outside the source tree. Heavy training
and real model reload/evaluation are separate validation steps.

## Long-context paired recall follow-up

The [long-recall protocol](../experiments/configs/reviewer/LONG_RECALL_PROTOCOL.txt)
defines four additional cohorts: classification-only and matching-index auxiliary
training, each through sequence length 1024 or 2048, evaluated through 8192.
Every cohort contains all three fixed seeds and four scaling/mean-pooling variants.
These new outputs are separate from the existing release catalog.

From a checkout with the experiment dependencies installed, use fresh output
directories (the runners refuse to overwrite existing runs):

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python experiments/runners/reviewer_long_recall.py --config experiments/configs/reviewer/long_recall_1024.json --output results/reviewer/long_recall_1024 --device cuda:0
python experiments/runners/reviewer_long_recall.py --config experiments/configs/reviewer/long_recall_2048.json --output results/reviewer/long_recall_2048 --device cuda:0
python experiments/runners/reviewer_long_recall_routing.py --config experiments/configs/reviewer/long_recall_routing_1024.json --output results/reviewer/long_recall_routing_1024 --device cuda:0
python experiments/runners/reviewer_long_recall_routing.py --config experiments/configs/reviewer/long_recall_routing_2048.json --output results/reviewer/long_recall_routing_2048 --device cuda:0
python experiments/runners/summarize_long_recall.py
python experiments/runners/validate_long_recall.py --device cuda:0
```

Add `--smoke` and use another output directory for an inexpensive implementation
check. The summary writes all per-seed results, aggregates, figures, and a
LaTeX result table under `results/reviewer/long_recall_summary/`. The audit
checks every stored prediction and freshly replays the first test batch at each
length for every checkpoint; it does not claim a full independent test replay.
Focused tests are in `tests/test_long_recall.py`.

Only `strict_frobenius` uses the strict denominator `N*24^2`. The `general_alpha1`
variant belongs to the general-scaling setting and carries the explicit large
factor `kappa=N*24^2`. Auxiliary training uses extra matching-index supervision;
inference receives only the input rows. Report all seeds and both protocols, and
keep recall performance separate from any claim of quantum computational advantage.

## Comment 4: native-temperature cap comparison

The [native-cap protocol](../experiments/configs/reviewer/NATIVE_RECALL_CAP_PROTOCOL.txt)
adds a classical comparison of caps 24 and 64 with native attention temperature
`sqrt(32)`. It uses the same three seeds, training batches and budget, and the
same disclosed matching-index auxiliary supervision. Cap 64 is a larger norm
allowance; its successes are not FN24 recoveries. Strict-scaling and separately
trained uniform controls are retained at both caps.

Run the frozen `reviewer_long_recall_routing.py` entrypoint with
`long_recall_native64_1024.json` and `long_recall_native64_2048.json`, writing to
`results/reviewer/long_recall_native64_1024` and its 2048 counterpart. Then reuse
the saved cap 24 checkpoints on the follow-up's fresh test stream:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python experiments/runners/reevaluate_long_recall_caps.py --train-length 1024 --output results/reviewer/long_recall_native24_1024 --device cuda:0
python experiments/runners/reevaluate_long_recall_caps.py --train-length 2048 --output results/reviewer/long_recall_native24_2048 --device cuda:0
python experiments/runners/validate_native_recall_caps.py --device cuda:0
python experiments/runners/summarize_native_recall_caps.py
```

The reevaluation requires the original `long_recall_routing_1024` and 2048
checkpoints, and performs no training. The audit checks every stored prediction
and freshly replays the first batch at each length. The summary writes the cap
comparison, all per-seed results and an exportable figure under
`results/reviewer/comment4_native_caps/`.
