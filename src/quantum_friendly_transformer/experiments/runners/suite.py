import argparse
import shlex
import subprocess
import sys
from pathlib import Path


DEFAULT_PYTHON = sys.executable
DEFAULT_CACHE = None


def parse_args():
    parser = argparse.ArgumentParser(description="Run or print the NLP revision experiment suite.")
    parser.add_argument("--profile", choices=["smoke", "pilot", "pilot-recovery", "minimum", "minimum-recovery"], default="minimum")
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE)
    parser.add_argument("--output-root", default="experiments/outputs")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--execute", action="store_true", help="Run commands. Without this flag, only print them.")
    parser.add_argument("--offline", action="store_true", help="Pass offline mode to HF-backed commands.")
    return parser.parse_args()


def q(command):
    return " ".join(shlex.quote(str(part)) for part in command)


def base_args(args):
    out = ["--device", args.device]
    if args.cache_dir:
        out.extend(["--cache-dir", args.cache_dir])
    if args.offline:
        out.append("--offline")
    return out


def smoke_commands(args):
    return [
        [args.python, "-m", "quantum_friendly_transformer.experiments.runners.diagnostics.smoke_local_models",
         "--output", f"{args.output_root}/raw/smoke_local_diagnostics.parquet"],
        [
            args.python,
            "-m", "quantum_friendly_transformer.experiments.runners.reports.aggregate_diagnostics",
            "--input-glob",
            f"{args.output_root}/raw/smoke_local_diagnostics.parquet",
            "--output-dir",
            f"{args.output_root}/aggregate_smoke",
        ],
        [
            args.python,
            "-m", "quantum_friendly_transformer.experiments.runners.reports.make_figures",
            "--aggregate-dir",
            f"{args.output_root}/aggregate_smoke",
            "--output-dir",
            f"{args.output_root}/figures_smoke",
        ],
        [
            args.python,
            "-m", "quantum_friendly_transformer.experiments.runners.reports.build_resource_table",
            "--aggregate-dir",
            f"{args.output_root}/aggregate_smoke",
            "--output",
            f"{args.output_root}/tables_smoke/resource_table_R1.csv",
        ],
    ]


def diagnostic_commands(args, pilot=False):
    raw = f"{args.output_root}/raw"
    shared = base_args(args)
    if pilot:
        return [
            [
                args.python,
                "-m", "quantum_friendly_transformer.experiments.runners.diagnostics.forward_stats",
                "--model",
                "bert-base-uncased",
                "--dataset",
                "glue",
                "--task",
                "sst2",
                "--split",
                "validation",
                "--max-samples",
                "16",
                "--seq-lens",
                "64",
                "--batch-size",
                "4",
                "--sample-rows-per-layer-head",
                "256",
                "--output",
                f"{raw}/pilot_diag_bert_sst2.parquet",
                *shared,
            ],
            [
                args.python,
                "-m", "quantum_friendly_transformer.experiments.runners.diagnostics.forward_stats",
                "--model",
                "gpt2",
                "--dataset",
                "wikitext",
                "--config",
                "wikitext-103-raw-v1",
                "--split",
                "validation",
                "--max-tokens",
                "8192",
                "--seq-lens",
                "128",
                "--batch-size",
                "2",
                "--sample-rows-per-layer-head",
                "256",
                "--output",
                f"{raw}/pilot_diag_gpt2_wikitext.parquet",
                *shared,
            ],
        ]
    return [
        [
            args.python,
            "-m", "quantum_friendly_transformer.experiments.runners.diagnostics.forward_stats",
            "--model",
            "bert-base-uncased",
            "--dataset",
            "glue",
            "--task",
            "sst2",
            "--split",
            "validation",
            "--max-samples",
            "1000",
            "--seq-lens",
            "128",
            "256",
            "512",
            "--batch-size",
            "16",
            "--output",
            f"{raw}/diag_bert_sst2.parquet",
            *shared,
        ],
        [
            args.python,
            "-m", "quantum_friendly_transformer.experiments.runners.diagnostics.forward_stats",
            "--model",
            "roberta-base",
            "--dataset",
            "glue",
            "--task",
            "mrpc",
            "--split",
            "validation",
            "--max-samples",
            "1000",
            "--seq-lens",
            "128",
            "256",
            "512",
            "--batch-size",
            "16",
            "--output",
            f"{raw}/diag_roberta_mrpc.parquet",
            *shared,
        ],
        [
            args.python,
            "-m", "quantum_friendly_transformer.experiments.runners.diagnostics.forward_stats",
            "--model",
            "gpt2",
            "--dataset",
            "wikitext",
            "--config",
            "wikitext-103-raw-v1",
            "--split",
            "validation",
            "--max-tokens",
            "200000",
            "--seq-lens",
            "512",
            "1024",
            "--batch-size",
            "4",
            "--output",
            f"{raw}/diag_gpt2_wikitext.parquet",
            *shared,
        ],
    ]


def normalisation_commands(args, pilot=False):
    raw = f"{args.output_root}/raw"
    shared = base_args(args)
    if pilot:
        return [
            [
                args.python,
                "-m", "quantum_friendly_transformer.experiments.runners.normalisation.glue",
                "--model",
                "bert-base-uncased",
                "--task",
                "sst2",
                "--variants",
                "baseline",
                "fn:4",
                "fn:8",
                "--epochs",
                "0.05",
                "--batch-size",
                "16",
                "--learning-rate",
                "2e-5",
                "--max-train-samples",
                "128",
                "--max-eval-samples",
                "128",
                "--output",
                f"{raw}/pilot_norm_bert_sst2.json",
                *shared,
            ],
            [
                args.python,
                "-m", "quantum_friendly_transformer.experiments.runners.normalisation.wikitext",
                "--model",
                "gpt2",
                "--variants",
                "baseline",
                "fn:4",
                "fn:8",
                "--eval-only",
                "--seq-len",
                "128",
                "--batch-size",
                "2",
                "--max-eval-tokens",
                "8192",
                "--output",
                f"{raw}/pilot_norm_gpt2_wikitext.json",
                *shared,
            ],
        ]
    commands = []
    for task, epochs, batch_size in [("sst2", "3", "32"), ("mrpc", "5", "16"), ("qnli", "3", "32"), ("rte", "5", "16")]:
        commands.append(
            [
                args.python,
                "-m", "quantum_friendly_transformer.experiments.runners.normalisation.glue",
                "--model",
                "bert-base-uncased",
                "--task",
                task,
                "--variants",
                "baseline",
                "sn:1",
                "fn:1",
                "fn:2",
                "fn:4",
                "fn:8",
                "--projection",
                "after_step",
                "--epochs",
                epochs,
                "--batch-size",
                batch_size,
                "--learning-rate",
                "2e-5",
                "--output",
                f"{raw}/norm_bert_{task}.json",
                *shared,
            ]
        )
    commands.append(
        [
            args.python,
            "-m", "quantum_friendly_transformer.experiments.runners.normalisation.wikitext",
            "--model",
            "gpt2",
            "--variants",
            "baseline",
            "fn:1",
            "fn:2",
            "fn:4",
            "fn:8",
            "--eval-only",
            "--seq-len",
            "1024",
            "--batch-size",
            "4",
            "--output",
            f"{raw}/norm_gpt2_wikitext.json",
            *shared,
        ]
    )
    return commands


def recovery_normalisation_commands(args, pilot=True):
    raw = f"{args.output_root}/raw"
    shared = base_args(args)
    prefix = "pilot_recovery" if pilot else "recovery"
    if pilot:
        return [
            [
                args.python,
                "-m", "quantum_friendly_transformer.experiments.runners.normalisation.glue",
                "--model",
                "bert-base-uncased",
                "--task",
                "sst2",
                "--variants",
                "baseline",
                "fn:8",
                "fn:16",
                "fn:32",
                "--projection",
                "after_step",
                "--epochs",
                "0.5",
                "--batch-size",
                "16",
                "--learning-rate",
                "2e-5",
                "--max-train-samples",
                "1024",
                "--max-eval-samples",
                "512",
                "--output",
                f"{raw}/{prefix}_norm_bert_sst2.json",
                *shared,
            ],
            [
                args.python,
                "-m", "quantum_friendly_transformer.experiments.runners.normalisation.wikitext",
                "--model",
                "gpt2",
                "--variants",
                "baseline",
                "fn:8",
                "fn:16",
                "fn:32",
                "fn:64",
                "--train-steps",
                "200",
                "--seq-len",
                "128",
                "--batch-size",
                "2",
                "--learning-rate",
                "5e-5",
                "--max-train-tokens",
                "32768",
                "--max-eval-tokens",
                "32768",
                "--output",
                f"{raw}/{prefix}_norm_gpt2_wikitext.json",
                *shared,
            ],
        ]
    return [
        [
            args.python,
            "-m", "quantum_friendly_transformer.experiments.runners.normalisation.wikitext",
            "--model",
            "gpt2",
            "--variants",
            "baseline",
            "fn:8",
            "fn:16",
            "fn:32",
            "fn:64",
            "--train-steps",
            "2000",
            "--seq-len",
            "512",
            "--batch-size",
            "2",
            "--learning-rate",
            "5e-5",
            "--max-train-tokens",
            "524288",
            "--max-eval-tokens",
            "131072",
            "--output",
            f"{raw}/{prefix}_norm_gpt2_wikitext.json",
            *shared,
        ]
    ]


def aggregate_commands(args, pilot=False, recovery=False):
    aggregate = f"{args.output_root}/aggregate"
    figures = f"{args.output_root}/figures"
    tables = f"{args.output_root}/tables"
    diag_glob = f"{args.output_root}/raw/{'pilot_' if pilot else ''}diag_*.parquet"
    if recovery:
        norm_prefix = "pilot_recovery_" if pilot else "recovery_"
    else:
        norm_prefix = "pilot_" if pilot else ""
    norm_glob = f"{args.output_root}/raw/{norm_prefix}norm_*.json"
    return [
        [args.python, "-m", "quantum_friendly_transformer.experiments.runners.reports.aggregate_diagnostics", "--input-glob", diag_glob, "--output-dir", aggregate],
        [args.python, "-m", "quantum_friendly_transformer.experiments.runners.reports.aggregate_normalisation", "--input-glob", norm_glob, "--output-dir", aggregate],
        [args.python, "-m", "quantum_friendly_transformer.experiments.runners.reports.make_figures", "--aggregate-dir", aggregate, "--output-dir", figures],
        [
            args.python,
            "-m", "quantum_friendly_transformer.experiments.runners.reports.build_resource_table",
            "--aggregate-dir",
            aggregate,
            "--output",
            f"{tables}/resource_table_R1.csv",
        ],
    ]


def build_commands(args):
    if args.profile == "smoke":
        return smoke_commands(args)
    if args.profile == "pilot-recovery":
        return recovery_normalisation_commands(args, pilot=True) + aggregate_commands(args, pilot=True, recovery=True)
    if args.profile == "minimum-recovery":
        return recovery_normalisation_commands(args, pilot=False) + aggregate_commands(args, pilot=False, recovery=True)
    pilot = args.profile == "pilot"
    return diagnostic_commands(args, pilot=pilot) + normalisation_commands(args, pilot=pilot) + aggregate_commands(args, pilot=pilot)


def main():
    args = parse_args()
    commands = build_commands(args)
    Path(args.output_root).mkdir(parents=True, exist_ok=True)
    for index, command in enumerate(commands, start=1):
        print(f"[{index}/{len(commands)}] {q(command)}", flush=True)
        if args.execute:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
