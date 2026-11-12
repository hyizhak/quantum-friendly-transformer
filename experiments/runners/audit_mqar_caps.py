"""Verify a changed-cap MQAR cohort against the frozen cap64 reference."""

import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/"src")]

import torch

from experiments.lib.reviewer import save_json
from experiments.runners.audit_mqar import audit_run
from experiments.runners.reviewer_mqar import load_data


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results",type=Path,default=ROOT/"results/reviewer/mqar_cap48")
    parser.add_argument("--assets",type=Path,default=ROOT/"results/reviewer/mqar_cap48_data")
    parser.add_argument("--reference-results",type=Path,default=ROOT/"results/reviewer/mqar_variance_init")
    parser.add_argument("--reference-assets",type=Path,default=ROOT/"results/reviewer/mqar_variance_init_data")
    parser.add_argument("--device",default="cuda:1")
    args=parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    config=json.loads((args.assets/"config.json").read_text())
    reference=json.loads((args.reference_assets/"config.json").read_text())
    differences={k for k in set(config)|set(reference) if config.get(k)!=reference.get(k)}
    if differences!={"cap","protocol"}:
        raise AssertionError(f"Unexpected comparison changes: {differences}")
    datasets=json.loads((args.assets/"manifest.json").read_text())["datasets"]
    for record in datasets:
        a=load_data(args.assets,record["name"])
        b=load_data(args.reference_assets,record["name"])
        if not torch.equal(a["inputs"],b["inputs"]) or not torch.equal(a["labels"],b["labels"]):
            raise AssertionError(f"Different data: {record['name']}")
    reports=[]
    for seed in config["seeds"]:
        starts=[]
        for variant in config["variants"]:
            run=args.results/f"{variant}_seed{seed}"
            result=json.loads((run/"result.json").read_text())
            old=json.loads((args.reference_results/run.name/"result.json").read_text())
            if result["smoke"]:
                raise AssertionError("Smoke result in benchmark cohort")
            if result["initial_state_sha256"]!=old["initial_state_sha256"]:
                raise AssertionError("Underlying initial parameters differ across caps")
            if result["training_steps"]!=old["training_steps"]:
                raise AssertionError("Training budgets differ")
            starts.append(result["initial_state_sha256"])
            reports.append(audit_run(run,args.assets,torch.device(args.device)))
            if reports[-1]["capped_matrices"]!=12:
                raise AssertionError("Unexpected capped-matrix coverage")
            print(f"PASS cap{config['cap']:g} {run.name}",flush=True)
        if len(set(starts))!=1:
            raise AssertionError("Unpaired native and uniform initial parameters")
    report={"status":"passed","models":len(reports),"cap":config["cap"],"reference_cap":reference["cap"],
        "identical_data":True,"matched_datasets":len(datasets),"identical_raw_initial_parameters":True,
        "effective_initial_weights_may_differ_due_to_cap":True,"same_training_budget":True,
        "configuration_differences":sorted(differences),
        "all_stored_queries_audited":sum(c["stored_predictions_audited"] for r in reports for c in r["conditions"]),
        "fresh_queries_replayed":sum(c["fresh_queries_replayed"] for r in reports for c in r["conditions"]),
        "replay_scope":"First batch at every condition for every model; not a second full test evaluation",
        "reports":reports}
    save_json(args.results/"audit.json",report)
    print(json.dumps({k:v for k,v in report.items() if k!="reports"},indent=2))


if __name__=="__main__":
    main()
