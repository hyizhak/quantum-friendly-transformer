# Library guide

## Installation and boundaries

The distribution is named `quantum-friendly-transformer`; Python imports use
`quantum_friendly_transformer`. Python 3.10+ and PyTorch 2.4+ (below 3.0) are
required. Version 0.3.0 also packages the NLP experiment runners, configurations,
recorded result tables, and an API for the full evidence archive. Attention
formulas and recorded experiment results are unchanged.

From a checkout, install with `python -m pip install .`. For development, use
`python -m pip install -e '.[dev]'`. A built wheel can be shared directly:

```bash
python -m pip install /path/to/quantum_friendly_transformer-0.3.0-py3-none-any.whl
```

These commands do not assume a PyPI publication.

| Extra | Purpose |
|---|---|
| `hf` | Transformers 4.x, einops, and safetensors; custom BERT and callbacks |
| `experiments` | `hf` plus datasets, metrics, plotting, tabular data, and configuration tools |
| `test` | pytest and the build frontend |
| `dev` | Experiment and test dependencies, plus distribution metadata validation |

Quantized/LoRA experiment modes additionally require their backend packages
(`bitsandbytes`/`peft`) on compatible hardware. They are not core dependencies.
`reqs/requirements.txt` is the historical research environment. `pyproject.toml`
defines library installation requirements; `uv.lock` records a resolved
development environment for users of uv.

## Public API

| Import | Contents |
|---|---|
| `quantum_friendly_transformer` | Version, caps, affine constraints, attention adapters, and statistics |
| `quantum_friendly_transformer.models` | PyTorch reference blocks and sequence/token classifiers |
| `quantum_friendly_transformer.data` | Tokenization, vocabulary, and collation helpers |
| `quantum_friendly_transformer.reproducibility` | `manual_seed` for Python, PyTorch, and NumPy when installed |
| `quantum_friendly_transformer.trainer` | `train` and `evaluate` for the reference classifiers |
| `quantum_friendly_transformer.norm_transformer.bert` | Optional Frobenius-normalized BERT configuration and models |
| `quantum_friendly_transformer.trainer.huggingface` | Optional `PrintMetricsCallback` |
| `quantum_friendly_transformer.trainer.metrics` | `compute_metrics`, using the `experiments` dependencies |

The original `norm_transformer.*`, `trainer.util`, and `util` imports remain
available. Implementations are not duplicated by public exports. Importing the
core library does not load Transformers, datasets, checkpoints, or GPU kernels.

## Caps and normalization

```python
import torch
from torch import nn
from quantum_friendly_transformer import UnitNormLayerNorm, learnable_frobenius_cap

projection = learnable_frobenius_cap(nn.Linear(16, 16), max_gamma=2.0)
model = nn.Sequential(projection, UnitNormLayerNorm(16))
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
output = model(torch.randn(2, 8, 16))
```

Install adapters before constructing the optimizer. `learnable_frobenius_cap`
parametrizes one weight as `min(1, gamma / ||W||_F) * W`, with trainable
`0 < gamma < max_gamma`. It leaves weights below the current gamma unchanged.
In contrast, `frobenius_norm` and `frobenius_norm_with_scaling` normalize every
selected weight, including weights already below the target norm.

For your loaded Transformer backbone:

```python
from quantum_friendly_transformer import (
    apply_learnable_frobenius_caps,
    collect_learnable_cap_stats,
    collect_weight_stats,
)

before = collect_weight_stats(model)
count = apply_learnable_frobenius_caps(model, max_gamma=24.0)
after = collect_weight_stats(model)
caps = collect_learnable_cap_stats(model)
```

Targets include recognized attention Q/K/V/output and MLP projections.
Embeddings and task heads are not selected by the default filter. Inspect
`count` and the returned matrix names; unrecognized backbones need a custom
`target_filter` or individual `learnable_frobenius_cap` calls.

`UnitNormLayerNorm` and `UnitNormRMSNorm` fix the affine gain to `1/sqrt(d)` and
the bias to zero. `apply_normalization_affine_constraints(model)` applies these
constraints to supported normalization modules. LayerNorm centers its input;
RMSNorm does not. With nonzero epsilon, output L2 norms can be below one.

## Attention scaling

`apply_attention_scaling` currently supports GPT-Neo, OLMo/OLMo2, and Mistral
attention modules with the attributes expected by the adapter. Native mode is
a no-op. Other modes fail if no supported modules are found; inspect the returned
records to verify coverage of every intended attention layer in mixed models.

```python
from quantum_friendly_transformer import apply_attention_scaling

records = apply_attention_scaling(
    model,
    "alpha_s_alpha_w",
    alpha_s=alpha_s_bound,
    alpha_w=alpha_w_bound,
)
```

Supply the bounds from your experiment design. This mode replaces the native
score multiplier with `1 / (alpha_s**2 * alpha_w**2)`. It does not calculate or
certify the input-matrix norm bound. A measured maximum token norm is not itself
a bound on the Frobenius norm of the full sequence matrix.

For trainable scaling, pass `trainable=True`. OLMo/Mistral need eager attention
when training the scale. Add `enforce_alpha_denominator_bound=True` to constrain
the learned denominator to remain at least the configured alpha denominator.
Call `project_attention_scales_(model, optimizer)` after optimizer steps for the
bounded controller's boundary/momentum handling. Calibration and theorem
validation remain the caller's responsibility.

## Saving and loading

Use `state_dict` checkpoints. Reconstruct the same architecture and install the
same adapters, caps, fixed normalization modules, and attention-controller
configuration before loading:

```python
torch.save(model.state_dict(), "model.pt")
# Recreate `restored` and apply the same adapters first.
restored.load_state_dict(torch.load("model.pt", map_location="cpu", weights_only=True))
```

Hook-based attention adapters are runtime objects; saving a state dict does not
install the hooks in another model. Bounded controller loading also validates
the configured denominator bound against the saved bound.

## Optional BERT models

Install `.[hf]`, then construct a local model without downloading weights:

```python
from quantum_friendly_transformer.norm_transformer.bert import (
    FrobeniuslyNormalizedBertConfig,
    FrobeniuslyNormalizedBertModel,
)

config = FrobeniuslyNormalizedBertConfig(
    vocab_size=32,
    hidden_size=16,
    num_hidden_layers=1,
    num_attention_heads=4,
    intermediate_size=32,
)
model = FrobeniuslyNormalizedBertModel(config)
```

The existing BERT implementation uses its PyTorch attention fallback. Triton is
not required. The base model returns encoded states and an optional pooled output.

## Research recipes

NLP recovery, partition-function and normalization diagnostics, depth
measurements, and report generation are installed under
`quantum_friendly_transformer.experiments`. Use `qft-experiment --list` for
entrypoints and `qft-evidence list` for the recorded artifacts. See the
[evidence guide](evidence.md) for package contents and companion archive access.

Fixed dataset/checkpoint/output recipes live in `experiments/runners/legacy/`.
The historical `src/quantum_friendly_transformer/trainer/*.py` entrypoints
forward there from a source checkout. An installed wheel contains only the
compatibility entrypoints; invoking a recipe explains how to obtain the checkout.
Reusable `train` and `evaluate` work independently of the checkout.

```bash
python experiments/runners/training.py --list-recipes
python experiments/runners/training.py --config experiments/configs/training/legacy_trainers.yaml
```

Run these commands from the repository root. They list or resolve recipes; add
`--execute` to launch a configured run. Checkpoints, GPUs, dataset downloads,
and local paths belong to the research protocol rather than the library install.

## Build and validate

```bash
python -m pip install -e '.[dev]'
python -m pytest
python -m compileall src alpha_s alpha_w experiments
python -m build
python -m twine check dist/*.whl dist/*.tar.gz
python examples/library_quickstart.py
```

`python -m build` creates a source archive and builds the wheel from that archive.
The version is defined once in `src/quantum_friendly_transformer/_version.py`.
Build isolation does not require importing PyTorch. Metadata, extras, and package
discovery use [setuptools' pyproject configuration](https://setuptools.pypa.io/en/latest/userguide/pyproject_config.html).

Core tests need no GPU, datasets, or model downloads:

```bash
python -m pytest tests/test_library.py tests/test_distribution.py tests/test_normalization.py tests/test_attention_scaling.py
```

The optional BERT smoke test skips when its dependencies are absent. NLP runner
tests additionally need the `experiments` dependencies. The source archive
includes core and research tests, both guides, the CPU example, and the evidence
builder alongside package code and compact evidence. Raw data and checkpoint
contents remain in the companion archive. Build commands do not publish them.
