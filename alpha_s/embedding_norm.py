"""Plot raw input-token embedding norms from a pinned local checkpoint."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from safetensors import safe_open


EMBEDDING_KEYS = ('model.embed_tokens.weight', 'transformer.wte.weight', 'wte.weight',
                  'transformer.tokens_embed.weight', 'tokens_embed.weight',
                  'bert.embeddings.word_embeddings.weight', 'roberta.embeddings.word_embeddings.weight')


def plot_embedding_norms(snapshot, output, *, model_label):
    candidates = []
    for path in Path(snapshot).glob('*.safetensors'):
        with safe_open(str(path), framework='pt', device='cpu') as reader:
            for key in EMBEDDING_KEYS:
                if key in reader.keys():
                    values = torch.linalg.vector_norm(reader.get_tensor(key).double(), dim=1).numpy()
                    candidates.append((key, values))
    if len(candidates) != 1:
        raise ValueError('Expected exactly one supported input embedding table')
    _, values = candidates[0]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(values, bins=30, edgecolor='black')
    ax.set(xlabel=r'Token embedding $\ell_2$ norm', ylabel='Number of vocabulary entries',
           title=f'{model_label}: raw input embedding table')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--model-label', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    plot_embedding_norms(args.snapshot, args.output, model_label=args.model_label)


if __name__ == '__main__':
    main()
