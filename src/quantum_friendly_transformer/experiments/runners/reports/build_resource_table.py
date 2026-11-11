import argparse
from pathlib import Path

from quantum_friendly_transformer.experiments.lib.common import ensure_parent


def parse_args():
    parser = argparse.ArgumentParser(description="Build empirical Resource Table R1.")
    parser.add_argument("--aggregate-dir", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def proxy(row):
    alpha_s = row.get("median_hidden_fro_over_sqrtN")
    overhead = row.get("softmax_overhead_from_p5_r") or row.get("softmax_overhead_p95")
    d = row.get("hidden_dim")
    if alpha_s != alpha_s or overhead != overhead or d != d:
        return None
    # Oracle-level constant proxy. alpha_w and alpha_m are reported separately when available.
    return float(alpha_s) * float(overhead) * float(d)


def main():
    args = parse_args()
    import pandas as pd

    constants_path = Path(args.aggregate_dir) / "resource_constants.csv"
    if not constants_path.exists():
        raise FileNotFoundError(f"missing {constants_path}; run experiments/runners/reports/aggregate_diagnostics.py first")
    constants = pd.read_csv(constants_path)
    rows = []
    for _, row in constants.iterrows():
        measured_alpha_s = row.get("median_hidden_fro_over_sqrtN")
        measured_alpha_w = row.get("realised_alpha_w", float("nan"))
        measured_alpha_m = row.get("realised_alpha_m", float("nan"))
        rows.append(
            {
                "model config": row.get("model"),
                "N": row.get("seq_len"),
                "d": row.get("hidden_dim"),
                "layers": row.get("num_layers"),
                "attention type": row.get("attention_type"),
                "normalisation type": row.get("norm_type"),
                "measured alpha_s / sqrt(N)": measured_alpha_s,
                "measured alpha_w": measured_alpha_w,
                "measured alpha_m": measured_alpha_m,
                "median Z/|A|": row.get("r_median"),
                "p5 Z/|A|": row.get("r_p5"),
                "max softmax overhead proxy over p5 tokens": row.get("softmax_overhead_from_p5_r"),
                "p5 LN/RMS denominator": row.get("normalisation_denom_p5"),
                "hidden-state norm growth factor": row.get("growth_F"),
                "oracle-query proxy": proxy(row),
                "input loading assumption": "QRAM/block-encoding oracle available; not costed here",
                "output extraction cost": "Not included in oracle-level proxy",
                "main caveat": "Measured constants only; not a hardware-level fault-tolerant resource estimate",
            }
        )
    output = ensure_parent(args.output)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
