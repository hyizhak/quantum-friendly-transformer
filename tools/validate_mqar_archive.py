"""Extract a source ZIP and run MQAR preparation, training, audits, and reports.

The default budget checks the workflow on CPU, retaining the published model
dimensions, seeds, variants, vocabulary, and test conditions. It is not a
reproduction of benchmark scores. Use --full-budget for the original recipes.
Existing Python dependencies are reused; the project itself is installed from
the extracted ZIP, and every experiment runs from that extracted source.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile


COHORTS = (
    ("initial", "mqar.py", "mqar.json"),
    ("variance_init", "mqar_variance_init.py", "mqar_variance_init.json"),
    ("standard", "mqar_standard.py", "mqar_standard.json"),
    ("cap48", "mqar_cap.py", "mqar_cap48.json"),
    ("cap24", "mqar_cap24.py", "mqar_cap24.json"),
)


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def equivalent_records(actual, expected):
    """Allow floating-point rounding across Python statistics implementations."""
    if isinstance(expected, dict):
        return (isinstance(actual, dict) and actual.keys() == expected.keys()
                and all(equivalent_records(actual[k], v) for k, v in expected.items()))
    if isinstance(expected, list):
        return (isinstance(actual, list) and len(actual) == len(expected)
                and all(equivalent_records(a, b) for a, b in zip(actual, expected)))
    if isinstance(expected, float):
        return isinstance(actual, float) and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-15)
    return actual == expected


def validate(args):
    destination = args.output.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    logs = destination / "logs"
    logs.mkdir()
    report_path = destination / "validation.json"
    report = {"status": "running", "archive": str(args.archive.resolve()),
              "archive_sha256": digest(args.archive), "full_training_budget": args.full_budget,
              "scope": "MQAR workflow validation using existing Python dependencies", "stages": []}

    def save():
        report_path.write_text(json.dumps(report, indent=2) + "\n")

    save()
    try:
        extracted = destination / "source"
        with zipfile.ZipFile(args.archive) as archive:
            for name in archive.namelist():
                path = Path(name)
                if path.is_absolute() or ".." in path.parts:
                    raise ValueError(f"Unsafe archive member: {name}")
            archive.extractall(extracted)
        candidates = ([extracted] if (extracted / "pyproject.toml").is_file()
                      else [p.parent for p in extracted.glob("*/pyproject.toml")])
        if len(candidates) != 1:
            raise ValueError("Expected one source tree containing pyproject.toml")
        source, = candidates
        if any(source.rglob(".git")):
            raise ValueError("The source ZIP must not contain Git metadata")
        report["source"] = str(source)
        report["git_metadata_present"] = False
        environment = dict(os.environ, HF_HUB_OFFLINE="1", MPLBACKEND="Agg",
                           MPLCONFIGDIR=str(destination / "plot-cache"),
                           PIP_DISABLE_PIP_VERSION_CHECK="1", PIP_NO_INDEX="1",
                           OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads))

        def run(name, command):
            print(f"START {name}", flush=True)
            started = time.monotonic()
            log = logs / f"{name}.log"
            stage = {"name": name, "command": [str(v) for v in command], "log": str(log)}
            report["stages"].append(stage)
            save()
            with log.open("w") as stream:
                completed = subprocess.run(stage["command"], cwd=source, env=environment,
                                           stdout=stream, stderr=subprocess.STDOUT)
            stage.update(exit_code=completed.returncode, seconds=time.monotonic() - started)
            save()
            if completed.returncode:
                raise RuntimeError(f"{name} failed; see {log}\n{log.read_text()[-4000:]}")
            print(f"PASS {name} ({stage['seconds']:.1f}s)", flush=True)

        def command(script, *arguments):
            return [sys.executable, source / "experiments/runners" / script, *arguments]

        installed = destination / "installed"
        run("install", [sys.executable, "-m", "pip", "install", "--no-deps", "--no-build-isolation",
                        "--target", installed, source])
        check = (
            "import sys, runpy; from pathlib import Path; "
            f"sys.path.insert(0, {str(installed)!r}); "
            "import quantum_friendly_transformer as qft; "
            f"assert Path(qft.__file__).is_relative_to(Path({str(installed)!r})); "
            f"runpy.run_path({str(source / 'examples/library_quickstart.py')!r}, run_name='__main__')"
        )
        run("installed_library", [sys.executable, "-I", "-c", check])
        recorded = source / "artifacts"
        if (recorded / "results/inventory.json").is_file():
            run("verify_recorded_data", [sys.executable, source / "tools/pack_result_data.py",
                                       "--verify", recorded / "results"])
            regenerated = destination / "recorded-mqar"
            run("reaggregate_recorded_mqar", command("summarize_mqar_predictions.py",
                "--capped-results", recorded / "results/mqar_cap24.zip",
                "--standard-results", recorded / "results/mqar_standard.zip",
                "--output", regenerated))
            actual = json.loads((regenerated / "summary.json").read_text())
            expected = json.loads((recorded / "mqar/summary.json").read_text())
            if not equivalent_records(actual, expected):
                raise ValueError("Bundled MQAR summary does not match its saved predictions")
            inventory = json.loads((recorded / "results/inventory.json").read_text())
            report["bundled_records_verified"] = sum(len(c["members"]) for c in inventory["cohorts"])
            report["recorded_prediction_queries_checked"] = actual["prediction_queries_checked"]
            save()
        # Exercise the documented prepare command with its original dataset sizes.
        run("prepare_original_config", command("mqar.py", "prepare", "--output", destination / "original-data"))
        configs = destination / "configs"
        configs.mkdir()
        assets, results = {}, {}
        for name, runner, filename in COHORTS:
            config = json.loads((source / "experiments/configs/mqar" / filename).read_text())
            if not args.full_budget:
                config.update(train_examples=128, validation_examples=8, test_examples=8,
                              epochs=1, workflow_validation=True)
            config["threads"] = args.threads
            config_path = configs / filename
            config_path.write_text(json.dumps(config, indent=2) + "\n")
            assets[name] = destination / "data" / name
            results[name] = destination / "runs" / name
            run(f"prepare_{name}", command(runner, "prepare", "--config", config_path, "--output", assets[name]))
            for variant in config["variants"]:
                run(f"train_{name}_{variant}", command("run_mqar_cohort.py", "--runner",
                    source / "experiments/runners" / runner, "--config", config_path,
                    "--assets", assets[name], "--output", results[name],
                    "--variant", variant, "--device", args.device))
            if name in {"initial", "variance_init"}:
                audit = command("audit_mqar.py", "--results", results[name], "--assets", assets[name])
            elif name == "standard":
                audit = command("audit_mqar_standard.py", "--results", results[name], "--assets", assets[name],
                                "--reference-assets", assets["variance_init"])
            else:
                audit = command("audit_mqar_caps.py", "--results", results[name], "--assets", assets[name],
                                "--reference-assets", assets["variance_init"],
                                "--reference-results", results["variance_init"])
            run(f"audit_{name}", [*audit, "--device", args.device])
        summaries = destination / "reports"
        run("summary_initialization", command("summarize_mqar.py", "--initial-results", results["initial"],
            "--followup-results", results["variance_init"], "--output", summaries / "initialization"))
        run("summary_standard", command("summarize_mqar_comparison.py", "--standard-results", results["standard"],
            "--capped-results", results["variance_init"], "--output", summaries / "standard"))
        run("summary_cap48", command("summarize_mqar_caps.py", "--standard-results", results["standard"],
            "--reference-results", results["variance_init"], "--capped-results", results["cap48"],
            "--output", summaries / "cap48"))
        run("summary_cap_series", command("summarize_mqar_cap_series.py", "--standard-results", results["standard"],
            "--cap-results", "64", results["variance_init"], "--cap-results", "48", results["cap48"],
            "--cap-results", "24", results["cap24"], "--output", summaries / "cap_series"))
        audits = {name: json.loads((directory / "audit.json").read_text()) for name, directory in results.items()}
        assert all(a["status"] == "passed" for a in audits.values())
        manifests = list((destination / "data").rglob("manifest.json")) + list((destination / "runs").rglob("manifest.json"))
        for path in manifests:
            provenance = json.loads(path.read_text())
            assert provenance["git_commit"] is None
            for filename, expected in provenance["source_sha256"].items():
                assert Path(filename).is_relative_to(source)
                assert digest(Path(filename)) == expected
        summaries_found = list(summaries.rglob("summary.json"))
        assert len(summaries_found) == 4
        assert all(json.loads(p.read_text())["workflow_validation"] == (not args.full_budget) for p in summaries_found)
        report.update(status="passed", models=sum(a["models"] for a in audits.values()),
                      cohort_audits={name: a["status"] for name, a in audits.items()},
                      stored_queries_audited=sum(a["all_stored_queries_audited"] for a in audits.values()),
                      fresh_queries_replayed=sum(a["fresh_queries_replayed"] for a in audits.values()),
                      source_manifests_verified=len(manifests), summaries=len(summaries_found),
                      reports=str(summaries), device=args.device)
        save()
        print(json.dumps({k: v for k, v in report.items() if k != "stages"}, indent=2), flush=True)
    except Exception as error:
        report.update(status="failed", error=str(error))
        save()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--full-budget", action="store_true")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    validate(args)


if __name__ == "__main__":
    main()
