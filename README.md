# quantum-friendly-transformer

An installable PyTorch library for Frobenius caps, spectral/Frobenius normalization,
normalization affine constraints, and attention scaling. This repository also
contains the experiments studying how these constraints affect Transformer models.

## Library installation

Python 3.10 or newer is required. From a source checkout:

```bash
python -m pip install .
```

The core library depends only on PyTorch. Choose optional dependencies for your workflow:

```bash
python -m pip install '.[hf]'             # Hugging Face and custom BERT models
python -m pip install -e '.[experiments]'  # Packaged experiment runners
python -m pip install -e '.[dev]'          # Experiments, tests, and build tools
```

Install the PyTorch build appropriate for your CPU/CUDA environment first if you
need a particular accelerator version. The older pinned environment in `reqs/`
is retained for reproducing the original experiments.

## Library usage

```python
import torch
from torch import nn
from quantum_friendly_transformer import UnitNormLayerNorm, learnable_frobenius_cap

projection = learnable_frobenius_cap(nn.Linear(16, 16), max_gamma=2.0)
model = nn.Sequential(projection, UnitNormLayerNorm(16))
output = model(torch.randn(2, 8, 16))
```

Use `apply_learnable_frobenius_caps(model, max_gamma=24.0)` to adapt supported
Transformer attention and MLP projections. Normalization, capping, and attention
scaling are separate operations. The original `norm_transformer` import paths
remain supported.

See [the library guide](docs/library.md) for APIs, supported attention adapters,
checkpoint loading, optional integrations, and release validation. A CPU example
that needs no downloads is available at `examples/library_quickstart.py`.

| Location | Purpose |
|---|---|
| `src/quantum_friendly_transformer/` | Distributable library and compatibility entrypoints |
| `src/quantum_friendly_transformer/experiments/` | Packaged NLP research runners, configs, and reporting |
| `src/quantum_friendly_transformer/evidence/` | Evidence API, checksummed catalog, and bundled records/tables |
| `experiments/lib/`, `experiments/runners/`, and `experiments/configs/` | Source-checkout research experiments, compatibility scripts, and original recipes in `runners/legacy/` |
| `experiments/vendor/zoology/` | Pinned MQAR data generator, upstream provenance, and license |
| `artifacts/` | Completed experimental results, per-example predictions, figure source tables, and checksums |
| `alpha_s/`, `alpha_w/`, `results/` | Analysis scripts and reported artifacts |
| `tests/` | CPU unit tests and research-runner tests |

The wheel includes the library, NLP experiment runners, and 540 recorded result,
metadata, table, and report files. Large raw measurements and checkpoints are
organized in a separate companion archive, accessible through the same library.
See [experiments and evidence](docs/evidence.md) for the inventory, archive layout,
checkpoint semantics, and reproduction commands.

```bash
qft-evidence list --group sst2
qft-evidence extract --destination ./research
qft-experiment --list
qft-experiment recovery.dispatcher --config recovery/tinystories_sst2_fn24_bounded_alpha_attention.yaml
```

The dispatcher prints commands unless `--execute` is supplied. Bundled evidence
works without downloading models or running training.

The later research experiments cover long-context paired recall, MQAR with
ordinary Transformer and capped/uniform baselines, and cap24 recovery with
training through length 8192. Their code, fixed configurations, protocols,
auditors, and tests are versioned in this repository and run from a source
checkout. Start with the [research experiment map](experiments/README.md#experiment-map)
and [experiment guide](docs/experiments.txt). Completed result records and
predictions are committed under [artifacts/](artifacts/README.md). They include
all retained training seeds; model weights and raw tensor datasets are omitted.
The historical wheel evidence remains a separate snapshot.

The MQAR commands also work from an extracted source ZIP; Git metadata is
optional, and source-file checksums are retained in each run's manifest.

## Conda setup

To set up the environment, we recommend using a virtual environment or conda environment. The following instructions assume you are using a conda environment.

```
bash setup.sh
```

This updates the `q-transformer` environment to Python 3.11 and installs the
library with its experiment dependencies. `environment.yml` and
`reqs/requirements.txt` retain the historical Python 3.8 environment; they are
not the setup specification for the current library.

To access Hugging Face datasets, you need to set up your Hugging Face token. You can do this by running:

```
huggingface-cli login
```
or by setting the `HF_TOKEN` environment variable with your token.

## Matrix Norms in Transformers

Before experiments showcasing the capability of normalized transformers, we provide scripts to check the norms of pretrained models, check `alpha_*` files:

1. To check the norms of inputs S, in `alpha_s`:

    To see the plots, run `plot_data.ipynb`

    The data is generated by running:

    ``` 
    python alpha_s.py {model_name} {max_tokens} {batch_size} {random}
    ```

    where:

    model_name: ['bert', 'bart', 'roberta', 'distilgpt', 'gpt2', 'gpt', 'llama2-7b', 'tinyllama', 'mistral7b']
    max_tokens: maximum number of tokens, usually 512; -1 disables MMLU truncation
    and selects 512 as the maximum random sequence length.

    batch_size: positive integer; larger models will need a smaller batch size to deal with OOM issues.

    random: 1 or 0 as a boolean variable to run the random dataset (1) or the MMLU dataset (0)

    **MMLU data correction:** The original compressed-SVD loop repeatedly measured
    the first sample of each batch for Llama 2, TinyLlama, and Mistral. All eight
    models shown in the MMLU plots have now been remeasured on the same 1531
    validation questions, and all seven MMLU PDFs and notebook plots regenerated.
    The script computes an exact SVD for each sample, excludes padding from both
    norms, and records the number of tokens actually measured after truncation.
    Thus the columns are `(N, embedding_fro, first_layer_fro, embedding_spectral)`.
    The three Llama/Mistral runs use no truncation; the five smaller models use a
    512-token limit. This also corrects the earlier padding and length mismatch.

    [The measurement manifest](alpha_s/data_files/mmlu_remeasurement.json) records
    pinned model and dataset revisions, verified weight hashes, output checksums,
    and real-model comparisons between batch sizes 1 and 4. Mistral uses its
    historical tokenizer to match the original tokenization. The public Llama 2
    mirror has the same weight hashes as the original checkpoint. The ordered
    inputs are identified by the manifest's dataset revision and question checksum.
    Download the `cais/mmlu` validation questions separately, in the recorded
    alphabetical subject order; the upstream question text is not redistributed.
    **Random-data correction:** All eight plotted models were also remeasured
    for each length from 1 to 512. The earlier arrays contained 259 NaNs in the
    unplotted first-layer column. New runs use exact SVD, float64 reductions,
    and `--seed 0`; actual token IDs are saved in `*_random_inputs.json`.
    These are new reproducible draws because the original seeds/IDs were lost.
    See [random_remeasurement.json](alpha_s/data_files/random_remeasurement.json).
    The seven `random*.pdf` files and cached notebook outputs were regenerated.

    BART is also remeasured, but is not included in these figures. Its historical
    files had incompatible 3/5-column schemas. The replacement records measure
    the native BART **encoder**, including its correctly loaded shared embedding
    table; loading `bart-base` as a causal LM had left that table randomly
    initialized in the tested Transformers version. See
    [bart_remeasurement.json](alpha_s/data_files/bart_remeasurement.json).

    For an offline rerun from a pinned snapshot, run from `alpha_s`:

    ```bash
    python alpha_s.py mistral7b -1 1 0 \
      --model-path /path/to/pinned/model/snapshot \
      --questions-json /path/to/local/mmlu_validation_questions.json \
      --output-dir /path/to/new/results --device cuda:0 --first-two-layers
    ```

    The optional `--first-two-layers` flag is supported for the three Llama/Mistral
    models. Two decoder layers preserve the measured `hidden_states[0]` and
    `hidden_states[1]`, without running subsequent layers. Measurement and plotting
    reject nonfinite values in any of the four columns and check
    `embedding_fro / sqrt(N) <= embedding_spectral <= embedding_fro`.
    Passing this numerical check alone does not prove sample
    alignment; the regression tests also check distinct samples and partial batches.

2. Weight measurements use pinned local snapshots and the paper's `S @ W`
    convention: `W` has shape `[input_width, output_width]`. PyTorch Linear
    weights must be transposed; GPT Conv1D weights already use this layout.
    GQA key/value matrices have fewer output columns than the hidden width.
    Reductions use float64 and variance uses `ddof=1`, pooling all actual
    columns across Q/K/V and all layers. From the repository root:

    ```bash
    python alpha_w/wqkv_norm_per_layer.py --sources alpha_w/stats/column_sources.json
    python alpha_w/measure_weights.py --sources alpha_w/stats/matrix_paper_sources.json --device cuda:0
    ```

    Adjust `snapshot_path` entries to your cache location. The corrected all-layer
    results are in [qkv_stats_corrected.csv](alpha_w/stats/qkv_stats_corrected.csv).
    The old `qkv_stats.csv`, `qkv_stats_gpt_series.csv`, and other legacy model
    histograms remain historical artifacts; they are superseded for the models
    listed in the corrected table. The three additional Llama3/Qwen/Nemo rows
    from the old CSV have not been remeasured in this audit.

    [MatrixNorm.pdf](alpha_s/MatrixNorm.pdf) and `alpha_s/languagemodels.ipynb`
    now read [measured Q/K/V values](alpha_w/stats/matrix_norms_paper_models.json),
    with no handwritten norms. They use the same eight models and legend order
    as the MMLU/random figures, in six panels with linear axes and ordinary
    numeric tick labels. Six models' V spectral norms had incorrectly used Q.
    This figure measures the **first layer**, and its horizontal axis is hidden
    width `d`; it is separate from the all-layer column statistics. Its Llama2
    model is **chat**, whereas the paper column table uses **base**. Both public
    mirrors have weight hashes verified against their respective official files.

    Regenerate the figure from the stored measurements without loading models:

    ```bash
    python alpha_w/measure_weights.py --figure reproduced/MatrixNorm.pdf
    ```

    `matrix_norms.json` and `matrix_sources.json` retain the broader historical
    measurement cohort; `matrix_paper_sources.json` selects the eight plotted
    models, including DistilGPT2.

    To regenerate the first-layer column histogram:

    ```bash
    python alpha_w/wqkv_norm.py --snapshot /path/to/snapshot \
      --model-label Llama2-7B-chat --output /path/to/column_norms.pdf
    ```

### Runtime:

The recorded MMLU remeasurement used A100 GPUs with model caches on the data disk.
Use `--device cpu` for a CPU run or `--device cuda:0` for a GPU run. Runtime depends
on the model, sequence lengths, hardware, and whether the weights are already cached.

## NLP Experiments

The additional research experiment code is in `experiments/lib/`,
`experiments/runners/*.py`, and `experiments/configs/`.
It covers bounded SST-2 recovery, matched uniform controls, associative recall,
pretrained diagnostics, and selective-layer examples. See
[the experiment guide](docs/experiments.txt) for entrypoints and tests.
These additions run from a source checkout. The [recorded data](artifacts/README.md)
supports table reaggregation without training or downloading checkpoints.

The NLP runners and portable configurations are installed with the library:

```bash
qft-experiment recovery.dispatcher --config recovery/tinystories_sst2_fn24_strict_recovery.yaml
qft-experiment recovery.dispatcher --config recovery/tinystories_sst2_fn24_bounded_alpha_attention.yaml
```

The dispatcher prints commands by default. Add `--execute` to run a configuration.
Use `qft-experiment --list-configs` for all available configurations and
`qft-experiment --list` for diagnostics and report generators.

`qft-evidence extract --destination ./research` restores the compact result
records, tables, and reports included in the package. Raw
measurements and saved checkpoints require the companion evidence archive.
See [the evidence guide](docs/evidence.md) for Python access, verification,
full extraction, and recovery-report reproduction.

## Training Normalized Transformers

The November 2026 code audit corrected RoPE coordinate pairing, the reference
attention module's layout/mask/output-projection handling, zero/half-precision
normalization, and exact spectral projection. It also fixed shared embedding
instances across legacy variants, checkpoint weight mapping into normalized
parameters, attention-only versus FFN-only flags, the Genomic Benchmarks save
prefix, restoration of the best validation checkpoint, and GUE selection on
`dev` instead of `test`. The two DNABERT fine-tuning recipes now initialize the
normalized variants from the same pretrained backbone and classifier as their
baseline, instead of leaving them randomly initialized.

The non-TATA seed-42 rerun uses the corrected implementation. Other
legacy training tables have not been regenerated by this rerun. Historical
checkpoints also contain the old RoPE cache; loading
now rebuilds that deterministic cache with the corrected pairing. GLUE evaluation
loss now weights batches by their sample counts, and diagnostic aggregation
includes the second RMS normalization denominator. The complete audit scope and
verification record are in [code_audit.json](artifacts/code_audit.json).

The archived initial rerun protocol covers all eight affected legacy recipes
with five paired seeds (105 runs). Execution was narrowed to seed-42 non-TATA;
the full multi-seed plan is not a completed study. Fixed splits, source hashes,
checkpoint selection and completed follow-ups are documented in
[the rerun guide](experiments/rigorous_rerun/README.md).

The seed-42 non-TATA rerun and follow-up runners are included in the source.
They export metrics, per-example predictions and validation selection records
into local runtime directories. Raw training histories, prediction files and
run metadata remain local and are not part of this source release. The rerun
guide documents the architecture, split and checkpoint-selection rules.

The original experiments ran on a single NVIDIA A100 SXM4 GPU (40 GiB HBM2, driver 535.154.05, CUDA 12.2) paired with an AMD EPYC 7713 processor—of which 16 threads are allocated from the system’s 64-core/128-thread CPU (1.5–2.0 GHz base, boosting to ≈3.1 GHz; 32 KiB L1d/L1i, 512 KiB L2, 32 MiB L3 cache)—and 110 GiB of DDR4 RAM, all under Ubuntu 20.04 LTS.

After setting up the environment, configure output directories in the recipes
under `experiments/runners/legacy/`. The historical commands below continue to
work from an installed source checkout.

To train single-layer normalized transformers on the [Genomic Benchmarks non-tata promoter dataset](https://bmcgenomdata.biomedcentral.com/articles/10.1186/s12863-023-01123-8)(0-1 classification task), use:

```
python src/quantum_friendly_transformer/trainer/train_genomic_bench.py
```

This result is reported in our paper and key hyperparameters are:
- `--learning-rate`: `8e-4`
- `--num-epochs`: `100`
- `--early-stopping`: `10`
- `--weight-decay`: `0.01`
- `--lr-scheduler-type`: `cosine`

We noted that more intricate hyperparameter tuning may be needed for the normalized models, as the optimization is slightly off the sweet point of current optimizers. For more details, please refer to the codebase.

### Runtime:
The training of 3 single-layer transformers (vanilla, spectral normalized, frobenius normalized) on the Genomic Benchmarks dataset with given hyperparameters takes approximately 1 hour the specified machine.

To fine-tune the transformer on the same dataset with specific layer normalized, use:

```
python src/quantum_friendly_transformer/trainer/fine_tune_sn_model_genomic_bench.py
```

To train a frobenius normalized DNABert on the same dataset, use:

```
python src/quantum_friendly_transformer/trainer/fine_tune_multilayer_genomic_bench.py
```

This result is reported in our paper and key hyperparameters are:
- `--learning-rate`: `2e-4`
- `--num-epochs`: `200`
- `--early-stopping`: `25`
- `--weight-decay`: `0.01`
- `--lr-scheduler-type`: `cosine`
  
For more details, please refer to the codebase.

### Runtime:
The training of a multi-layer normalized Bert model on the Genomic Benchmarks dataset with given hyperparameters takes approximately 2 hour on the specified machine.

To train an one-layer transformer on the [Genome Understanding Evaluation (GUE) notata promoter dataset](https://huggingface.co/datasets/leannmlindsey/GUE)(0-1 classification task), use:

```
python src/quantum_friendly_transformer/trainer/train_gue.py
```

To fine-tune the transformer on the same dataset with specific layer normalized, use:

```
python src/quantum_friendly_transformer/trainer/fine_tune_sn_model_gue.py
```

To train a frobenius normalized DNABert on the same dataset, use:

```
python src/quantum_friendly_transformer/trainer/fine_tune_multilayer_gue.py
```

To train an one-layer transformer on the [conll 2003 dataset](https://huggingface.co/datasets/eriktks/conll2003)(POS classification task), use:

```
python src/quantum_friendly_transformer/trainer/train_conll2003.py
```

To fine-tune the transformer on the same dataset with specific layer normalized, use:

```
python src/quantum_friendly_transformer/trainer/fine_tune_sn_model_conll2003.py
```
