"""Verify the ordinary Transformer baseline and its matched MQAR datasets."""

import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/"src")]

import torch

from experiments.lib.mqar_standard import MQARStandardTransformer
from experiments.lib.reviewer import save_json
from experiments.runners import audit_mqar as audit
from experiments.runners.reviewer_mqar import load_data


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results",type=Path,default=ROOT/"results/reviewer/mqar_standard")
    parser.add_argument("--assets",type=Path,default=ROOT/"results/reviewer/mqar_standard_data")
    parser.add_argument("--reference-assets",type=Path,default=ROOT/"results/reviewer/mqar_variance_init_data")
    parser.add_argument("--device",default="cuda:1")
    args=parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    config=json.loads((args.assets/"config.json").read_text())
    reference=json.loads((args.reference_assets/"config.json").read_text())
    allowed={"cap","variants","initialization","protocol","layernorm_epsilon"}
    differences={key for key in set(config)|set(reference) if config.get(key)!=reference.get(key)}
    if differences-allowed:
        raise AssertionError(f"Unexpected protocol changes: {differences-allowed}")
    datasets=json.loads((args.assets/"manifest.json").read_text())["datasets"]
    for record in datasets:
        actual=load_data(args.assets,record["name"])
        previous=load_data(args.reference_assets,record["name"])
        if not torch.equal(actual["inputs"],previous["inputs"]) or not torch.equal(actual["labels"],previous["labels"]):
            raise AssertionError(f"Unmatched benchmark data: {record['name']}")
    audit.MQARTransformer=MQARStandardTransformer
    reports=[]
    for seed in config["seeds"]:
        run=args.results/f"native_seed{seed}"
        reports.append(audit.audit_run(run,args.assets,torch.device(args.device)))
        if reports[-1]["capped_matrices"]!=0:
            raise AssertionError("The ordinary baseline must have no matrix caps")
        print(f"PASS standard_seed{seed}",flush=True)
    report={"status":"passed","models":len(reports),"identical_reference_data":True,
            "matched_datasets":len(datasets),"allowed_configuration_differences":sorted(differences),
            "all_stored_queries_audited":sum(c["stored_predictions_audited"] for r in reports for c in r["conditions"]),
            "fresh_queries_replayed":sum(c["fresh_queries_replayed"] for r in reports for c in r["conditions"]),
            "replay_scope":"First batch at every condition for every model; not a second full test evaluation",
            "reports":reports}
    save_json(args.results/"audit.json",report)
    print(json.dumps({k:v for k,v in report.items() if k!="reports"},indent=2))


if __name__=="__main__":
    main()
