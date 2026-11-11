import argparse
import glob
import json
from pathlib import Path

from quantum_friendly_transformer.experiments.lib.common import ensure_parent


HIGHER_IS_BETTER = {
    "accuracy": True,
    "f1": True,
    "matthews": True,
    "pearson": True,
    "spearman": True,
    "eval_loss": False,
    "loss": False,
    "perplexity": False,
}


VARIANT_LABELS = {
    "baseline": "baseline",
    "sn:1": "SN_tau1",
    "sn:2": "SN_tau2",
    "sn:4": "SN_tau4",
    "sn:8": "SN_tau8",
    "fn:1": "FN_tau1",
    "fn:2": "FN_tau2",
    "fn:4": "FN_tau4",
    "fn:8": "FN_tau8",
    "fn:1+recovery": "FN_tau1_recovery",
    "fn:2+recovery": "FN_tau2_recovery",
    "fn:4+recovery": "FN_tau4_recovery",
    "fn:8+recovery": "FN_tau8_recovery",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate GLUE/WikiText normalisation outputs for Table B1.")
    parser.add_argument("--input-glob", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def variant_label(raw):
    if raw in VARIANT_LABELS:
        return VARIANT_LABELS[raw]
    base = raw.removesuffix("+recovery")
    recovery = raw.endswith("+recovery")
    if ":" in base:
        kind, tau = base.split(":", 1)
        if kind in {"fn", "sn"}:
            prefix = "FN" if kind == "fn" else "SN"
            tau_label = tau.replace(".", "p")
            return f"{prefix}_tau{tau_label}" + ("_recovery" if recovery else "")
    return raw.replace(":", "_").replace("+", "_")


def realised_constants(stats):
    if not stats:
        return None, None, None
    fro = [row.get("frobenius_norm") for row in stats if row.get("frobenius_norm") is not None]
    spec = [row.get("spectral_norm") for row in stats if row.get("spectral_norm") is not None]
    col = [row.get("max_col_norm") for row in stats if row.get("max_col_norm") is not None]
    return (max(fro) if fro else None, max(spec) if spec else None, max(col) if col else None)


def task_name(payload):
    if payload.get("task"):
        return payload["task"]
    if payload.get("dataset") == "wikitext":
        return payload.get("config", "wikitext")
    return payload.get("dataset", "unknown")


def read_rows(path):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    rows = []
    model = payload.get("model", "unknown")
    task = task_name(payload)
    dataset = payload.get("dataset", "unknown")
    for result in payload.get("results", []):
        raw_variant = result.get("variant", "unknown")
        label = variant_label(raw_variant)
        alpha_w, alpha_spectral, alpha_m = realised_constants(result.get("weight_stats_final", []))
        for metric, value in result.get("metrics", {}).items():
            if metric in {"tokens", "samples", "examples"}:
                continue
            if not isinstance(value, (int, float)):
                continue
            rows.append(
                {
                    "source_file": str(path),
                    "model": model,
                    "dataset": dataset,
                    "task": task,
                    "metric": metric,
                    "variant": raw_variant,
                    "variant_label": label,
                    "value": float(value),
                    "kind": result.get("kind"),
                    "tau": result.get("tau"),
                    "recovery": result.get("recovery", False),
                    "train_steps": result.get("train_steps"),
                    "projected_after_load": result.get("projected_after_load"),
                    "realised_alpha_w": alpha_w,
                    "realised_alpha_spectral": alpha_spectral,
                    "realised_alpha_m": alpha_m,
                }
            )
    return rows


def wide_table(pd, long_df):
    if long_df.empty:
        return long_df
    pivot = (
        long_df.pivot_table(
            index=["model", "dataset", "task", "metric"],
            columns="variant_label",
            values="value",
            aggfunc="first",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    fn_cols = [col for col in pivot.columns if col.startswith("FN_tau") and not col.endswith("recovery")]
    best_values = []
    best_labels = []
    drops = []
    for _, row in pivot.iterrows():
        metric = row["metric"]
        higher = HIGHER_IS_BETTER.get(metric, True)
        candidates = [(col, row[col]) for col in fn_cols if col in row and row[col] == row[col]]
        if not candidates:
            best_values.append(None)
            best_labels.append(None)
            drops.append(None)
            continue
        best_col, best_value = max(candidates, key=lambda item: item[1]) if higher else min(candidates, key=lambda item: item[1])
        baseline = row.get("baseline")
        if baseline == baseline:
            drop = (baseline - best_value) if higher else (best_value - baseline)
        else:
            drop = None
        best_values.append(best_value)
        best_labels.append(best_col)
        drops.append(drop)
    pivot["best_FN"] = best_values
    pivot["best_FN_variant"] = best_labels
    pivot["best_FN_drop"] = drops

    constants = []
    for _, row in pivot.iterrows():
        best_label = row.get("best_FN_variant")
        if not best_label:
            constants.append({"realised_alpha_w": None, "realised_alpha_spectral": None, "realised_alpha_m": None})
            continue
        match = long_df[
            (long_df["model"] == row["model"])
            & (long_df["dataset"] == row["dataset"])
            & (long_df["task"] == row["task"])
            & (long_df["metric"] == row["metric"])
            & (long_df["variant_label"] == best_label)
        ]
        if match.empty:
            constants.append({"realised_alpha_w": None, "realised_alpha_spectral": None, "realised_alpha_m": None})
        else:
            first = match.iloc[0]
            constants.append(
                {
                    "realised_alpha_w": first.get("realised_alpha_w"),
                    "realised_alpha_spectral": first.get("realised_alpha_spectral"),
                    "realised_alpha_m": first.get("realised_alpha_m"),
                }
            )
    constants_df = pd.DataFrame(constants)
    return pd.concat([pivot.reset_index(drop=True), constants_df], axis=1)


def main():
    args = parse_args()
    import pandas as pd

    files = [path for path in sorted(glob.glob(args.input_glob)) if not path.endswith(".metadata.json")]
    if not files:
        raise FileNotFoundError(f"no non-metadata files matched {args.input_glob!r}")
    rows = []
    for path in files:
        rows.extend(read_rows(path))
    long_df = pd.DataFrame(rows)
    if long_df.empty:
        raise RuntimeError("normalisation result files contained no metric rows")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(ensure_parent(output_dir / "table_B1_normalisation_long.csv"), index=False)
    wide_table(pd, long_df).to_csv(ensure_parent(output_dir / "table_B1_normalisation_wide.csv"), index=False)
    print(f"aggregated {len(files)} normalisation files into {output_dir}")


if __name__ == "__main__":
    main()
