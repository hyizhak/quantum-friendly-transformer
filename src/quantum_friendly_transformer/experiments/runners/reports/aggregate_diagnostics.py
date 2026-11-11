#!/usr/bin/env python
"""Aggregate raw forward diagnostic parquet files."""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

from quantum_friendly_transformer.experiments.lib.aggregation import ATTENTION_METRICS, NORM_METRICS, hidden_growth, quantile_table, resource_constants
from quantum_friendly_transformer.experiments.lib.common import ensure_parent


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-glob", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    import pandas as pd

    files = sorted(glob.glob(args.input_glob))
    if not files:
        raise FileNotFoundError(f"no files matched {args.input_glob!r}")
    df = pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    attention_groups = ["model", "dataset", "task", "seq_len", "layer"]
    norm_groups = ["model", "dataset", "task", "seq_len", "layer"]
    attention = quantile_table(df, attention_groups, ATTENTION_METRICS)
    norms = quantile_table(df, norm_groups, NORM_METRICS)
    growth = hidden_growth(df)
    resource = resource_constants(df, attention, norms, growth)

    if attention is not None:
        attention.to_csv(ensure_parent(output_dir / "table_A1_attention_quantiles.csv"), index=False)
    if norms is not None:
        norms.to_csv(ensure_parent(output_dir / "table_A1_norm_quantiles.csv"), index=False)
    growth.to_csv(ensure_parent(output_dir / "table_C1_hidden_growth.csv"), index=False)
    resource.to_csv(ensure_parent(output_dir / "resource_constants.csv"), index=False)
    print(f"aggregated {len(files)} files into {output_dir}")


if __name__ == "__main__":
    main()
