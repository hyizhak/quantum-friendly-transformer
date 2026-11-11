"""Optional Frobenius-normalized BERT models (install the ``hf`` extra)."""

from importlib import import_module

__all__ = [
    "FrobeniuslyNormalizedBertConfig",
    "FrobeniuslyNormalizedBertModel",
    "FrobeniuslyNormalizedBertForMaskedLM",
    "FrobeniuslyNormalizedBertForSequenceClassification",
]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        module = import_module(".frobenius_normalized_bert", __name__)
    except ModuleNotFoundError as exc:
        if exc.name in {"transformers", "einops"}:
            raise ImportError(
                "BERT models require optional dependencies. Install "
                "'quantum-friendly-transformer[hf]'."
            ) from exc
        raise
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
