"""Plot first-layer column norms in the S @ W matrix convention."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

if __package__:
    from .measure_weights import iter_qkv
else:
    from measure_weights import iter_qkv


def plot_column_norms(snapshot, output, *, model_label):
    layer, weights, _ = next(iter_qkv(snapshot, first_only=True))
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, kind in zip(axes, 'qkv'):
        values = torch.linalg.vector_norm(weights[kind].double(), dim=0).numpy()
        ax.hist(values, bins=30, edgecolor='black')
        ax.set(title=r'$W_' + kind + '$', xlabel=r'Column $\ell_2$ norm', ylabel='Frequency')
        # Autoscale: fixed limits previously hid legitimate large-norm columns.
    fig.suptitle(f'{model_label}: first-layer column norms ({layer})')
    fig.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--model-label', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    plot_column_norms(args.snapshot, args.output, model_label=args.model_label)


if __name__ == '__main__':
    main()
