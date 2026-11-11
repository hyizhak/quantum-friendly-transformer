"""Compatibility imports for the original research utilities.

New code can import tokenization from ``data``, RNG seeding from
``reproducibility``, and optional callbacks/metrics from ``trainer`` modules.
Hugging Face is loaded only when its callback is explicitly requested.
"""

import torch

from .data import (
    LetterTokenizer,
    build_vocab,
    coll_factory,
    get_idx_to_label,
    get_label_to_idx,
    token_to_idx,
    tokenize_and_align_labels,
    tokenize_dna_sequence_genomic_bench,
    tokenize_dna_sequence_gue,
)
from .reproducibility import manual_seed
from .trainer.metrics import compute_metrics

__all__ = [
    "LetterTokenizer", "build_vocab", "coll_factory", "get_idx_to_label",
    "get_label_to_idx", "token_to_idx", "tokenize_and_align_labels",
    "tokenize_dna_sequence_genomic_bench", "tokenize_dna_sequence_gue",
    "manual_seed", "compute_metrics", "PrintMetricsCallback", "check_matrix_norm",
]


def __getattr__(name):
    if name == "PrintMetricsCallback":
        from .trainer.huggingface import PrintMetricsCallback

        globals()[name] = PrintMetricsCallback
        return PrintMetricsCallback
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))


def check_matrix_norm(model_path):
    import numpy as np
    from safetensors import safe_open

    tensors = {}
    with safe_open(model_path, framework="pt", device="cpu") as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)

    gamma_list = []

    for name, param in tensors.items():
        if name.endswith(".g"):
            g_value = param.data.cpu()  # move to CPU if needed
            gamma_value = torch.sigmoid(g_value)
            gamma_list.append(gamma_value)
            print(f"Parameter: {name}")
            print(f"g values:\n{g_value}")
            print(f"gamma values:\n{gamma_value}\n")

    print(np.mean(gamma_list))
    print(np.var(gamma_list))
