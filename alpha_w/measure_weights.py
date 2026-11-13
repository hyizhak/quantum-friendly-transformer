"""Reproducible Q/K/V measurements from local Hugging Face snapshots.

All matrices use the paper's S @ W convention: [input_width, output_width].
Linear weights are transposed; GPT Conv1D weights already have this layout.
Only encoder self-attention is included for encoder-decoder models. ALBERT
shared parameters are counted once. No inference or model downloads occur.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

import torch


# Match the model cohort and ordering in the MMLU/random input-norm figures.
PAPER_MODELS = ("bert", "roberta", "distilgpt", "gpt2", "gpt",
                "llama2-7b", "tinyllama", "mistral7b")


_PATTERNS = (
    (r"((?:bert|roberta)\.encoder\.layer\.\d+)\.attention\.self\.(query|key|value)\.weight", "linear"),
    (r"(distilbert\.transformer\.layer\.\d+)\.attention\.(q_lin|k_lin|v_lin)\.weight", "linear"),
    (r"(albert\.encoder\.albert_layer_groups\.\d+\.albert_layers\.\d+)\.attention\.(query|key|value)\.weight", "linear"),
    (r"((?:model\.)?encoder\.layers\.\d+)\.self_attn\.(q_proj|k_proj|v_proj)\.weight", "linear"),
    (r"(encoder\.block\.\d+)\.layer\.0\.SelfAttention\.(q|k|v)\.weight", "linear"),
    (r"(model\.layers\.\d+)\.self_attn\.(q_proj|k_proj|v_proj)\.weight", "linear"),
    (r"((?:transformer\.)?h\.\d+)\.attn\.(c_attn)\.weight", "conv1d"),
)


def projection_descriptor(key):
    for pattern, layout in _PATTERNS:
        match = re.fullmatch(pattern, key)
        if match:
            layer, projection = match.groups()
            return layer, projection[0] if layout == "linear" else "qkv", layout
    return None


def canonical_projections(weight, projection, layout):
    if weight.ndim != 2:
        raise ValueError("Attention weights must be matrices")
    if layout == "linear" and projection in "qkv" and len(projection) == 1:
        return {projection: weight.T}
    if layout == "conv1d" and projection == "qkv":
        if weight.shape[1] != 3 * weight.shape[0]:
            raise ValueError("GPT Conv1D QKV weight must have shape [d, 3*d]")
        return dict(zip("qkv", weight.chunk(3, dim=1)))
    raise ValueError(f"Unsupported projection/layout: {projection}/{layout}")


def iter_qkv(snapshot, *, first_only=False):
    """Stream relevant safetensors, yielding each complete layer exactly once."""
    from safetensors import safe_open

    snapshot = Path(snapshot)
    records = {}
    for file in sorted(snapshot.glob("*.safetensors")):
        with safe_open(str(file), framework="pt", device="cpu") as reader:
            for key in reader.keys():
                descriptor = projection_descriptor(key)
                if descriptor is not None:
                    layer, projection, layout = descriptor
                    if (layer, projection) in records:
                        raise ValueError(f"Duplicate projection: {layer}/{projection}")
                    records[layer, projection] = (file, key, layout)
    if not records:
        raise ValueError(f"No supported self-attention weights in {snapshot}")
    layers = sorted({layer for layer, _ in records},
                    key=lambda name: tuple(map(int, re.findall(r"\d+", name))))
    for layer in layers[:1] if first_only else layers:
        weights, keys = {}, {}
        for (record_layer, projection), (file, key, layout) in records.items():
            if record_layer != layer:
                continue
            with safe_open(str(file), framework="pt", device="cpu") as reader:
                parts = canonical_projections(reader.get_tensor(key), projection, layout)
            for kind, value in parts.items():
                if kind in weights:
                    raise ValueError(f"Duplicate {kind} projection in {layer}")
                weights[kind], keys[kind] = value, key
        if set(weights) != set("qkv"):
            raise ValueError(f"Incomplete QKV projections in {layer}")
        yield layer, weights, keys


def matrix_norms(weight, *, device="cpu"):
    value = weight.detach().to(device=device, dtype=torch.float64)
    if not torch.isfinite(value).all():
        raise ValueError("Nonfinite weight matrix")
    frobenius = torch.linalg.vector_norm(value).item()
    spectral = torch.linalg.svdvals(value)[0].item()
    if spectral > frobenius * (1 + 1e-10):
        raise ValueError("Spectral norm exceeds Frobenius norm")
    return {"shape": list(weight.shape), "frobenius": frobenius, "spectral": spectral}


def column_statistics(snapshot):
    norms = {kind: [] for kind in "qkv"}
    shapes = []
    for layer, weights, keys in iter_qkv(snapshot):
        shapes.append({"layer": layer, "shapes": {k: list(v.shape) for k, v in weights.items()},
                       "source_keys": keys})
        for kind, weight in weights.items():
            norms[kind].append(torch.linalg.vector_norm(weight.double(), dim=0))
    norms = {kind: torch.cat(values) for kind, values in norms.items()}
    norms["total"] = torch.cat(list(norms.values()))
    stats = {"d_hidden": shapes[0]["shapes"]["q"][0], "layers": len(shapes)}
    for kind, values in norms.items():
        if not torch.isfinite(values).all():
            raise ValueError(f"Nonfinite {kind} column norms")
        prefix = "total" if kind == "total" else "W" + kind
        stats.update({prefix + "_mean": values.mean().item(),
                      prefix + "_var": values.var(correction=1).item(),
                      prefix + "_count": values.numel()})
    return stats, shapes


def measure_sources(sources, *, device="cpu"):
    models = {}
    for name, source in sources.items():
        layer, weights, keys = next(iter_qkv(source["snapshot_path"], first_only=True))
        print(f"Measuring {name}: {layer}", flush=True)
        projections = {kind: {"source_key": keys[kind], **matrix_norms(weights[kind], device=device)}
                       for kind in "qkv"}
        models[name] = {"source": source, "layer": layer,
                        "hidden_width": weights["q"].shape[0], "projections": projections}
    return {"schema_version": 1, "scope": "first encoder/self-attention layer",
            "matrix_convention": "S @ W; W has shape [input_width, output_width]",
            "method": "float64 Frobenius norm and exact SVD; no randomized approximation",
            "measurement_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "models": models}


def plot_matrix_norms(record, output, models=PAPER_MODELS):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FixedFormatter, MaxNLocator, NullFormatter, ScalarFormatter
    import seaborn as sns

    missing = set(models) - record["models"].keys()
    if missing:
        raise ValueError(f"Missing plotted model measurements: {sorted(missing)}")
    plt.rcParams.update({"font.size": 12})
    # Use the same six-panel canvas, default subplot positions, label sizes
    # and figure legend as alpha_s/plot_data.ipynb's combined plot. Matrix-norm
    # axes are linear, with ordinary numeric tick labels, as requested.
    # In particular, do not apply tight_layout: the input-norm figures do not.
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    colors = sns.color_palette("husl", n_colors=len(models))
    for col in range(3):
        axes[1, col].set_xlabel(r"Hidden width $d$", fontsize=16)
    axes[0, 0].set_ylabel("Frobenius Norm", fontsize=16)
    axes[1, 0].set_ylabel("Spectral Norm", fontsize=16)
    for row, norm in enumerate(("frobenius", "spectral")):
        for col, kind in enumerate("qkv"):
            ax = axes[row, col]
            ax.set_xscale("linear")
            ax.set_yscale("linear")
            widths = sorted({record["models"][name]["hidden_width"] for name in models})
            ax.xaxis.set_major_locator(FixedLocator(widths))
            ax.xaxis.set_major_formatter(FixedFormatter([str(width) for width in widths]))
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))
            ordinary_numbers = ScalarFormatter(useOffset=False)
            ordinary_numbers.set_scientific(False)
            ax.yaxis.set_major_formatter(ordinary_numbers)
            ax.yaxis.set_minor_formatter(NullFormatter())
            for i, name in enumerate(models):
                model = record["models"][name]
                measurement = model["projections"][kind]
                ax.scatter(model["hidden_width"], measurement[norm], s=36,
                           color=colors[i], marker="o",
                           label=name if row == 0 and col == 0 else None)
            ax.set_ylim(bottom=0)
            ax.text(.05, .93, r"$W_" + kind + "$", transform=ax.transAxes,
                    ha="left", va="top", fontsize=12)
            ax.legend([], [], frameon=False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncols=9, fontsize=12)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, help="JSON mapping names to pinned local model sources")
    parser.add_argument("--measurements", type=Path, default=Path(__file__).parent / "stats/matrix_norms_paper_models.json")
    parser.add_argument("--figure", type=Path, default=Path(__file__).resolve().parents[1] / "alpha_s/MatrixNorm.pdf")
    parser.add_argument("--models", nargs="+", default=list(PAPER_MODELS),
                        help="Plotted models; defaults to the eight MMLU/random models")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.sources:
        record = measure_sources(json.loads(args.sources.read_text()), device=args.device)
        args.measurements.parent.mkdir(parents=True, exist_ok=True)
        args.measurements.write_text(json.dumps(record, indent=2) + "\n")
    else:
        record = json.loads(args.measurements.read_text())
    plot_matrix_norms(record, args.figure, models=args.models)


if __name__ == "__main__":
    main()
