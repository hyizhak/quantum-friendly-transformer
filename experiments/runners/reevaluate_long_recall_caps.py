"""Reevaluate saved cap24 recall models on the cap64 follow-up's fresh test set.

This performs no training and never modifies the original run directories.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/"src")]

import torch

from experiments.runners import reviewer_long_recall as core
from experiments.lib.reviewer import manifest,save_json


def main(args):
    original=ROOT/f"results/reviewer/long_recall_routing_{args.train_length}"
    old_config=json.loads((original/"config.json").read_text())
    new_config=json.loads((ROOT/f"experiments/configs/reviewer/long_recall_native64_{args.train_length}.json").read_text())
    config=copy.deepcopy(old_config)
    config.update(test_seed=new_config["test_seed"],variants=new_config["variants"],protocol="saved_cap24_reevaluation_on_native_cap_followup_test")
    for key in ("dimension","key_dimension","ffn_dimension","curriculum","steps","batch_pairs","learning_rate","adam_epsilon","weight_decay","routing_auxiliary_weight","seeds","test_lengths","test_pairs","test_batch_pairs"):
        assert config[key]==new_config[key],key
    args.output.mkdir(parents=True,exist_ok=False)
    save_json(args.output/"config.json",config)
    plan=[{"seed":s,"variant":v,"status":"planned"} for s in config["seeds"] for v in config["variants"]]
    save_json(args.output/"plan.json",plan)
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    device=torch.device(args.device)
    sources=[Path(__file__),Path(core.__file__),ROOT/"experiments/lib/long_recall.py",
             ROOT/"experiments/lib/reviewer.py",ROOT/"src/quantum_friendly_transformer/norm_transformer/frobenius_cap.py",
             ROOT/"src/quantum_friendly_transformer/norm_transformer/normalization.py",
             ROOT/"experiments/configs/reviewer/NATIVE_RECALL_CAP_PROTOCOL.txt"]
    for item in plan:
        item["status"]="running"
        save_json(args.output/"plan.json",plan)
        parent=original/f"{item['variant']}_seed{item['seed']}"
        output=args.output/parent.name
        output.mkdir(exist_ok=False)
        parent_manifest=json.loads((parent/"manifest.json").read_text())
        for path,expected in parent_manifest["source_sha256"].items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==expected
        result=json.loads((parent/"result.json").read_text())
        assert hashlib.sha256((parent/"final.pth").read_bytes()).hexdigest()==result["checkpoint_sha256"]
        checkpoint=torch.load(parent/"final.pth",map_location=device,weights_only=True)
        assert checkpoint["config"]==old_config
        model=core.make_model(old_config,item["variant"],device)
        model.load_state_dict(checkpoint["state_dict"])
        provenance=manifest(config,sources,seed=item["seed"],variant=item["variant"],smoke=False,
            training_performed=False,training_config=old_config,
            parent_training_run=str(parent.relative_to(ROOT)),
            parent_manifest_sha256=hashlib.sha256((parent/"manifest.json").read_bytes()).hexdigest(),
            checkpoint_sha256=result["checkpoint_sha256"],device=str(device),evaluation_dtype="float64",
            selection="Reuse the original fixed-final checkpoint; no new training or selection")
        save_json(output/"running_manifest.json",provenance)
        save_json(output/"config.json",config)
        shutil.copyfile(parent/"final.pth",output/"final.pth")
        metrics=core.evaluate(model,config,output,device)
        summary=[]
        for n in config["test_lengths"]:
            batches=[r for r in metrics if r["N"]==n]
            count=sum(r["examples"] for r in batches)
            summary.append({"N":n,"examples":count,
                **{prefix+"accuracy":sum(r[prefix+"correct"] for r in batches)/count
                   for prefix in ["","uniform_","shuffled_query_"]+[f"noise_{e:g}_" for e in config["attention_noise_l2"]]}})
        result.update(evaluation=summary,reevaluation_only=True,parent_training_run=str(parent.relative_to(ROOT)))
        save_json(output/"result.json",result)
        save_json(output/"manifest.json",provenance)
        item["status"]="completed"
        save_json(args.output/"plan.json",plan)
        print(f"REEVALUATED {parent.name}",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-length",type=int,choices=[1024,2048],required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--device",default="cuda:0")
    main(parser.parse_args())
