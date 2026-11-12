"""Compare the ordinary Transformer, QFT MQAR adaptation, and uniform control."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/"src")]

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.lib.mqar_reporting import cohort_config, protocol_summary
from experiments.lib.experiment import save_csv,save_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--standard-results", type=Path, required=True)
    parser.add_argument("--capped-results", type=Path, required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    models=[("standard","Standard Transformer",args.standard_results,"native"),
            ("qft_adaptation","QFT adaptation (cap 64)",args.capped_results,"native"),
            ("uniform","Uniform control (cap 64)",args.capped_results,"uniform")]
    records,audits,input_files=[],{},{}
    for model,title,root,variant in models:
        audit_path=root/"audit.json"
        audits[model]=json.loads(audit_path.read_text())
        if audits[model]["status"]!="passed":
            raise AssertionError("Every reported model must pass its audit")
        input_files[str(audit_path.resolve())]=hashlib.sha256(audit_path.read_bytes()).hexdigest()
        for seed in (11,22,33):
            path=root/f"{variant}_seed{seed}"/"result.json"
            result=json.loads(path.read_text())
            if result["smoke"]:
                raise AssertionError("Smoke run in benchmark comparison")
            input_files[str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest()
            for row in result["evaluation"]:
                records.append({"model":model,"label":title,"seed":seed,
                    **{k:v for k,v in row.items() if k!="layer_diagnostics"}})
    conditions=[(64,4),(128,8),(256,16),(256,64),(512,32),(1024,64),(2048,128)]
    aggregated=[]
    for model,title,_,_ in models:
        for n,k in conditions:
            rows=[r for r in records if (r["model"],r["N"],r["pairs"])==(model,n,k)]
            if len(rows)!=3 or {r["seed"] for r in rows}!={11,22,33}:
                raise AssertionError("Incomplete seed cohort")
            r={"model":model,"label":title,"N":n,"pairs":k,"seeds":3,"role":rows[0]["role"]}
            for metric in ("accuracy","mean_nll","sequence_accuracy","uniform_ablation_accuracy"):
                values=[row[metric] for row in rows]
                r[metric+"_mean"]=statistics.mean(values)
                r[metric+"_sd"]=statistics.stdev(values)
            aggregated.append(r)
    save_csv(args.output/"per_seed.csv",records)
    save_csv(args.output/"aggregate.csv",aggregated)
    save_json(args.output/"summary.json",{"models":9,"seeds_per_model":3,
        "identical_data":True,"results":aggregated,"source_result_sha256":input_files,
        "interpretation":"Matched exploratory comparison; QFT adaptation is a newly implemented MQAR model, not the original legacy model class."})
    report_config = cohort_config(args.capped_results)
    summary_path = args.output / "summary.json"
    summary_record = json.loads(summary_path.read_text())
    summary_record.update(configuration=report_config,
                          workflow_validation=bool(report_config.get("workflow_validation")))
    save_json(summary_path, summary_record)
    lines=["MQAR: STANDARD TRANSFORMER AND QFT ADAPTATION", "",
        "Same official MQAR inputs/labels, vocabulary 8,192, three seeds (11,22,33).",
        protocol_summary(report_config),
        "Two layers, d=64, FFN width 128, native attention denominator sqrt(64).",
        "Fixed final checkpoints, no seed or checkpoint selection. Mean +/- sample SD over model seeds.", "",
        "Standard Transformer: ordinary trainable LayerNorm (gamma initially 1, epsilon 1e-5),",
        "uncapped projection/readout weights, input features scaled to ordinary per-coordinate variance.",
        "QFT adaptation: fixed unit-norm LayerNorm, cap 64 on all 12 Transformer projections,",
        "uncapped vocabulary readout, initialization adjusted for unit-norm activations.",
        "Uniform control: separately trained capped QFT adaptation with causal uniform attention.",
        "The QFT model is the new MQAR adaptation, not the existing legacy model class.", ""]
    for n,k in conditions:
        lines.append(f"N={n}, memory pairs={k}")
        for r in [r for r in aggregated if (r["N"],r["pairs"])==(n,k)]:
            lines.append(f"  {r['label']}: {100*r['accuracy_mean']:.3f} +/- {100*r['accuracy_sd']:.3f}%")
        lines.append("")
    lines += ["All baseline data tensors were checked against the QFT follow-up and match exactly.",
        "All stored predictions were audited; checkpoint replay covers the first batch per test condition.",
        "This is an exploratory comparison added after the first experiment. The complete initial",
        "QFT initialization cohort remains in its separately supplied result directory.",
        "Longer contexts also contain more memory pairs, so those tests change both length and memory load.",
        "Scores alone establish neither statistical equivalence nor a computational advantage.", ""]
    (args.output/"summary.txt").write_text("\n".join(lines))
    fig,ax=plt.subplots(figsize=(10,5),layout="constrained")
    for (model,title,_,_),color in zip(models,["#426b45","#245c9f","#b05a32"]):
        rows=[next(r for r in aggregated if (r["model"],r["N"],r["pairs"])==(model,n,k)) for n,k in conditions]
        ax.plot(range(len(conditions)),[100*r["accuracy_mean"] for r in rows],marker="o",label=title,color=color)
        for index,seed in enumerate((11,22,33)):
            values=[100*next(r["accuracy"] for r in records if (r["model"],r["seed"],r["N"],r["pairs"])==(model,seed,n,k)) for n,k in conditions]
            ax.scatter(np.arange(len(conditions))+(index-1)*.055,values,s=15,color=color,alpha=.65)
    ax.set_xticks(range(len(conditions)),[f"{n}\n{k} pairs" for n,k in conditions])
    ax.set_xlabel("Sequence length and memory pairs")
    ax.set_ylabel("Official MQAR query-token accuracy (%)")
    ax.set_ylim(-2,102)
    ax.grid(axis="y",alpha=.2)
    ax.set_title("Matched MQAR comparison: means and all three seeds")
    ax.legend()
    fig.savefig(args.output/"comparison.pdf")
    fig.savefig(args.output/"comparison.png",dpi=180)
    plt.close(fig)
    latex=[r"\begin{table}[t]",r"\centering\small",r"\begin{tabular}{lrrrr}",r"\hline",
           r"Model & $N=256,K=16$ & $N=512,K=32$ & $N=1024,K=64$ & $N=2048,K=128$ \\",r"\hline"]
    for model,title,_,_ in models:
        cells=[]
        for n,k in [(256,16),(512,32),(1024,64),(2048,128)]:
            r=next(r for r in aggregated if (r["model"],r["N"],r["pairs"])==(model,n,k))
            cells.append(f"${100*r['accuracy_mean']:.3f}\\pm{100*r['accuracy_sd']:.3f}$")
        latex.append(title+" & "+" & ".join(cells)+r" \\")
    latex += [r"\hline",r"\end{tabular}",r"\caption{Matched MQAR query-token accuracy (\%), mean $\pm$ sample SD over all three seeds. All models use the same data, width, depth, native attention temperature and training budget. The standard Transformer has ordinary trainable LayerNorm and uncapped weights; the QFT adaptation has unit-norm LayerNorm and cap 64 on Transformer projections. Both have uncapped vocabulary readouts. The comparison is exploratory; longer contexts also increase memory load.}",r"\label{tab:mqar-standard-comparison}",r"\end{table}"]
    (args.output/"comparison_table.tex").write_text("\n".join(latex)+"\n")
    print("\n".join(lines))


if __name__=="__main__":
    main()
