"""Audit the Comment 4 native-temperature cap comparison and replay test batches."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/"src")]

import torch
from experiments.lib.long_recall import input_certificate,matrix_records,paired_batch
from experiments.runners.reviewer_long_recall import make_model,state_hash


def read_csv(path):
    with path.open(newline="") as stream: return list(csv.DictReader(stream))


@torch.inference_mode()
def validate(root,device):
    reports,initials,labels_by_length,configs={},{},{},{}
    for cap in (24,64):
        for train_n in (1024,2048):
            directory=root/f"long_recall_native{cap}_{train_n}"
            config=json.loads((directory/"config.json").read_text())
            configs[cap,train_n]=config
            assert config["cap"]==cap and config["test_seed"]==1279933 and config["seeds"]==[11,22,33]
            plan=json.loads((directory/"plan.json").read_text())
            assert len(plan)==9 and all(x["status"]=="completed" for x in plan)
            for item in plan:
                run=directory/f"{item['variant']}_seed{item['seed']}"
                manifest=json.loads((run/"manifest.json").read_text())
                result=json.loads((run/"result.json").read_text())
                assert manifest["config"]==config==json.loads((run/"config.json").read_text())
                for path,expected in manifest["source_sha256"].items():
                    assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==expected,path
                assert hashlib.sha256((run/"final.pth").read_bytes()).hexdigest()==result["checkpoint_sha256"]
                checkpoint=torch.load(run/"final.pth",map_location=device,weights_only=True)
                assert checkpoint["seed"]==item["seed"] and checkpoint["variant"]==item["variant"]
                training_config=checkpoint["config"]
                if cap==24:
                    assert manifest["training_performed"] is False and result["reevaluation_only"] is True
                    parent=ROOT/manifest["parent_training_run"]
                    assert hashlib.sha256((parent/"manifest.json").read_bytes()).hexdigest()==manifest["parent_manifest_sha256"]
                    assert hashlib.sha256((parent/"final.pth").read_bytes()).hexdigest()==result["checkpoint_sha256"]
                    assert training_config==manifest["training_config"]
                else:
                    assert training_config==config
                    history=read_csv(run/"training.csv")
                    assert int(history[-1]["step"])==3000
                    for r in read_csv(run/"weights_training.csv"):
                        assert float(r["effective_frobenius"])<=cap*(1+1e-7)
                torch.manual_seed(item["seed"])
                model=make_model(training_config,item["variant"],device)
                initial=state_hash(model.state_dict())
                assert initial==result["initial_state_sha256"]
                assert initials.setdefault(item["seed"],initial)==initial
                model.load_state_dict(checkpoint["state_dict"])
                assert state_hash(model.state_dict())==state_hash(checkpoint["state_dict"])
                model.double().eval()
                weights=matrix_records(model)
                assert len(weights)==6 and max(x["effective_frobenius"] for x in weights)<=cap*(1+1e-7)
                for k in ("training_input_row_squared_max","training_input_matrix_squared_to_N_max"):
                    assert 0<result[k]<=1
                predictions=read_csv(run/"predictions.csv")
                metrics=read_csv(run/"test.csv")
                assert len(predictions)==24576
                max_error,replayed=0.,0
                for evaluation in result["evaluation"]:
                    n=evaluation["N"]
                    rows=[r for r in predictions if int(r["N"])==n]
                    batches=[r for r in metrics if int(r["N"])==n]
                    assert len(rows)==evaluation["examples"]==4096
                    signature=hashlib.sha256(bytes(int(r["label"]) for r in rows)).hexdigest()
                    assert labels_by_length.setdefault(n,signature)==signature
                    for i,r in enumerate(rows):
                        assert int(r["example"])==i and int(r["pair"])==i//2
                        if i%2:
                            assert int(r["label"])==1-int(rows[i-1]["label"])
                            assert r["uniform_prediction"]==rows[i-1]["uniform_prediction"]
                    for prefix in ["","uniform_","shuffled_query_"]+[f"noise_{e:g}_" for e in config["attention_noise_l2"]]:
                        correct=sum(r[prefix+"prediction"]==r["label"] for r in rows)
                        assert correct==sum(int(r[prefix+"correct"]) for r in batches)
                        assert correct/len(rows)==evaluation[prefix+"accuracy"]
                    assert evaluation["uniform_accuracy"]==.5
                    if item["variant"]=="uniform": assert evaluation["accuracy"]==.5
                    for r in batches:
                        assert float(r["maximum_input_row_squared_norm"])<=1
                        assert float(r["maximum_input_matrix_squared_norm"])<=n
                        assert float(r["maximum_paired_mean_gap"])==0
                        assert min(float(r["minimum_sigma1"]),float(r["minimum_sigma2"]))>0
                        if item["variant"]=="strict_frobenius": assert float(r["maximum_absolute_logit"])<=1/n+1e-12
                    generator=torch.Generator(device=device).manual_seed(config["test_seed"]+n)
                    inputs,labels,selected=paired_batch(config["test_batch_pairs"],n,config["dimension"],config["key_dimension"],generator,device)
                    input_certificate(inputs)
                    inputs=inputs.double()
                    logits,diag=model(inputs,diagnostics=True)
                    assert torch.equal(inputs[::2].mean(1),inputs[1::2].mean(1))
                    index=torch.arange(len(labels),device=device)
                    margins=logits[index,labels]-logits[index,1-labels]
                    for i,r in enumerate(rows[:len(labels)]):
                        assert int(r["prediction"])==int(logits[i].argmax())
                        assert int(r["label"])==int(labels[i])
                        error=abs(float(r["margin"])-float(margins[i]))
                        max_error=max(max_error,error)
                        assert error<1e-8
                    direct,hidden,_=model.dense_forward(inputs[:4])
                    assert float((direct-logits[:4]).abs().max())<1e-9
                    assert float((hidden-diag["hidden"][:4]).abs().max())<1e-9
                    replayed+=len(labels)
                key=f"{directory.name}/{run.name}"
                reports[key]={"cap":cap,"new_training":cap==64,"stored_predictions":len(predictions),
                    "fresh_replay_predictions":replayed,"maximum_replay_margin_error":max_error,
                    "checkpoint_sha256":result["checkpoint_sha256"]}
                print(f"PASS {key}",flush=True)
    for n in (1024,2048):
        old={k:v for k,v in configs[24,n].items() if k not in ("cap","protocol")}
        new={k:v for k,v in configs[64,n].items() if k not in ("cap","protocol")}
        assert old==new,"Unmatched cap comparison"
    output=root/"comment4_native_caps"
    output.mkdir(exist_ok=True)
    payload={"all_checks_passed":True,"models":len(reports),"new_training_runs":18,"saved_checkpoints_reevaluated":18,
        "stored_predictions_audited":sum(r["stored_predictions"] for r in reports.values()),
        "fresh_predictions_replayed":sum(r["fresh_replay_predictions"] for r in reports.values()),
        "same_initial_weights_by_seed":True,"same_training_protocol_except_cap":True,"same_fresh_test_stream":True,
        "scope":"All stored predictions audited; first batch at every length freshly replayed. Cap24 checkpoints were fully reevaluated, not retrained.",
        "reports":reports}
    (output/"audit.json").write_text(json.dumps(payload,indent=2)+"\n")
    print(json.dumps({k:v for k,v in payload.items() if k!="reports"},indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results",type=Path,default=ROOT/"results/reviewer")
    parser.add_argument("--device",default="cuda:0")
    args=parser.parse_args()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    validate(args.results,torch.device(args.device))
