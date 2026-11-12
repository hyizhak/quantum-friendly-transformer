"""Summarize completed MQAR cap cohorts without rewriting earlier reports."""

import argparse
import csv
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

from experiments.lib.reviewer import save_csv,save_json


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--caps",type=int,nargs="+",default=[64,48,24])
    parser.add_argument("--output",type=Path,default=ROOT/"results/reviewer/mqar_cap24_comparison")
    args=parser.parse_args()
    if len(args.caps)!=len(set(args.caps)) or any(cap<=0 for cap in args.caps):
        raise ValueError("Supply distinct positive caps")
    models=[("standard","Standard Transformer",ROOT/"results/reviewer/mqar_standard","native",None)]
    roots={cap:ROOT/("results/reviewer/mqar_variance_init" if cap==64 else f"results/reviewer/mqar_cap{cap}") for cap in args.caps}
    models += [(f"cap{cap}",f"QFT adaptation (cap {cap})",roots[cap],"native",cap) for cap in args.caps]
    models += [(f"uniform{cap}",f"Uniform control (cap {cap})",roots[cap],"uniform",cap) for cap in args.caps]
    records,norms,inputs=[],[],{}
    conditions=[(64,4),(128,8),(256,16),(256,64),(512,32),(1024,64),(2048,128)]
    for model,title,root,variant,cap in models:
        audit_path=root/"audit.json"
        audit=json.loads(audit_path.read_text())
        if audit["status"]!="passed":
            raise AssertionError(f"Audit not passed: {root}")
        inputs[str(audit_path.relative_to(ROOT))]=file_hash(audit_path)
        for seed in (11,22,33):
            folder=root/f"{variant}_seed{seed}"
            result_path=folder/"result.json"
            result=json.loads(result_path.read_text())
            config_path=folder/"config.json"
            config=json.loads(config_path.read_text())
            if result["smoke"] or result["seed"]!=seed or result["variant"]!=variant:
                raise AssertionError("Unexpected result identity")
            if config["cap"]!=cap:
                raise AssertionError("Cap label differs from actual run configuration")
            for path in (result_path,config_path):
                inputs[str(path.relative_to(ROOT))]=file_hash(path)
            for row in result["evaluation"]:
                records.append({"model":model,"label":title,"cap":cap,"seed":seed,
                    **{k:v for k,v in row.items() if k!="layer_diagnostics"}})
            for stage in ("initial","final"):
                path=folder/f"weights_{stage}.csv"
                inputs[str(path.relative_to(ROOT))]=file_hash(path)
                with path.open() as stream:
                    targets=[r for r in csv.DictReader(stream) if r["capped"]=="True"]
                if cap is not None and (len(targets)!=12 or any(float(r["effective_frobenius"])>cap*(1+1e-6) for r in targets)):
                    raise AssertionError("Incorrect cap coverage or an exceeded bound")
                norms.append({"model":model,"seed":seed,"stage":stage,"cap":cap,
                    "capped_matrices":len(targets),"active_caps":sum(r["cap_active"]=="True" for r in targets),
                    "max_effective_capped_norm":max((float(r["effective_frobenius"]) for r in targets),default=None),
                    "max_raw_capped_norm":max((float(r["raw_frobenius"]) for r in targets),default=None),
                    "active_matrix_names":",".join(r["matrix"] for r in targets if r["cap_active"]=="True")})
    aggregated=[]
    for model,title,_,_,cap in models:
        for n,k in conditions:
            rows=[r for r in records if (r["model"],r["N"],r["pairs"])==(model,n,k)]
            if len(rows)!=3 or {r["seed"] for r in rows}!={11,22,33}:
                raise AssertionError("Incomplete seed cohort")
            row={"model":model,"label":title,"cap":cap,"N":n,"pairs":k,"seeds":3,"role":rows[0]["role"]}
            for metric in ("accuracy","mean_nll","sequence_accuracy","uniform_ablation_accuracy"):
                row[metric+"_mean"]=statistics.mean(r[metric] for r in rows)
                row[metric+"_sd"]=statistics.stdev(r[metric] for r in rows)
            aggregated.append(row)
    args.output.mkdir(parents=True,exist_ok=True)
    save_csv(args.output/"per_seed.csv",records)
    save_csv(args.output/"aggregate.csv",aggregated)
    save_csv(args.output/"cap_activity.csv",norms)
    save_json(args.output/"summary.json",{"caps":args.caps,"total_models":len(models)*3,
        "all_seeds_retained":True,"results":aggregated,"cap_activity":norms,
        "source_result_sha256":inputs,"scope":"Exploratory matched cap comparison of the MQAR adaptation; every cohort trained from scratch."})
    lines=["MQAR CAP SERIES: "+", ".join(map(str,args.caps)),"",
        "Same official MQAR data, raw random initial parameters, native attention denominator sqrt(64),",
        "unit-norm LayerNorm, optimizer, minibatch orders and 32-epoch training budget across capped models.",
        "All three seeds (11,22,33); fixed final checkpoints; answer-token supervision only.",
        "Training: N=256, 16 memory pairs, 20,000 sequences, vocabulary 8,192.",
        "Testing: 1,000 shared sequences per condition; full-vocabulary query-token accuracy.",
        "All 12 complete Transformer projection matrices capped; the vocabulary readout remains uncapped.",
        "Effective initial weights differ when a smaller cap clips the same raw initial parameters.",
        "These are MQAR adaptations, not the legacy model class or pretrained SST-2 recovery runs.",
        "Mean +/- sample standard deviation across the three training seeds follows.",""]
    for n,k in conditions:
        lines.append(f"N={n}, memory pairs={k}")
        for row in [r for r in aggregated if (r["N"],r["pairs"])==(n,k)]:
            lines.append(f"  {row['label']}: {100*row['accuracy_mean']:.3f} +/- {100*row['accuracy_sd']:.3f}%")
        lines.append("")
    lines += ["Primary condition: individual attention-model seeds"]
    for row in [r for r in records if (r["N"],r["pairs"])==(256,16) and not r["model"].startswith("uniform")]:
        lines.append(f"  {row['label']} seed{row['seed']}: {100*row['accuracy']:.5f}%")
    lines += ["","Final native-attention cap activity"]
    for row in [r for r in norms if r["stage"]=="final" and r["model"].startswith("cap")]:
        lines.append(f"  {row['model']} seed{row['seed']}: {row['active_caps']}/{row['capped_matrices']} active; max effective norm {row['max_effective_capped_norm']:.4f}; {row['active_matrix_names']}")
    lines += ["","Interpretation limits:",
        "Cap choices were requested after earlier results. These are exploratory comparisons using the",
        "same test stream, not newly untouched confirmatory tests. Every seed and earlier cohort is retained.",
        "Longer contexts also increase memory load. Failure under a fixed training recipe does not prove",
        "architectural impossibility. Successful MQAR scores do not establish quantum advantage or",
        "recovery of pretrained model quality. Cap24 here refers to this synthetic benchmark adaptation.",""]
    (args.output/"summary.txt").write_text("\n".join(lines))
    fig,ax=plt.subplots(figsize=(10,5),layout="constrained")
    plotted=[("standard","Standard Transformer","#426b45")]
    colors=["#245c9f","#9c3d70","#d18a23","#6655a0","#329294"]
    plotted += [(f"cap{cap}",f"QFT adaptation: cap {cap}",colors[i%len(colors)]) for i,cap in enumerate(args.caps)]
    for model,title,color in plotted:
        rows=[next(r for r in aggregated if (r["model"],r["N"],r["pairs"])==(model,n,k)) for n,k in conditions]
        ax.plot(range(len(conditions)),[100*r["accuracy_mean"] for r in rows],marker="o",color=color,label=title)
        for i,seed in enumerate((11,22,33)):
            values=[100*next(r["accuracy"] for r in records if (r["model"],r["seed"],r["N"],r["pairs"])==(model,seed,n,k)) for n,k in conditions]
            ax.scatter(np.arange(len(conditions))+(i-1)*.055,values,s=15,color=color,alpha=.65)
    ax.set_xticks(range(len(conditions)),[f"{n}\n{k} pairs" for n,k in conditions])
    ax.set_xlabel("Sequence length and memory pairs")
    ax.set_ylabel("Official MQAR query-token accuracy (%)")
    ax.set_ylim(-2,102)
    ax.grid(axis="y",alpha=.2)
    ax.set_title("MQAR cap series: means and all three seeds")
    ax.legend()
    fig.savefig(args.output/"cap_series.pdf")
    fig.savefig(args.output/"cap_series.png",dpi=180)
    plt.close(fig)
    latex=[r"\begin{table}[t]",r"\centering\small",r"\begin{tabular}{lrrrr}",r"\hline",
        r"Model & $N=256,K=16$ & $N=512,K=32$ & $N=1024,K=64$ & $N=2048,K=128$ \\",r"\hline"]
    for model,title,_,_,_ in models:
        cells=[]
        for n,k in [(256,16),(512,32),(1024,64),(2048,128)]:
            r=next(r for r in aggregated if (r["model"],r["N"],r["pairs"])==(model,n,k))
            cells.append(f"${100*r['accuracy_mean']:.3f}\\pm{100*r['accuracy_sd']:.3f}$")
        latex.append(title+" & "+" & ".join(cells)+r" \\")
    latex += [r"\hline",r"\end{tabular}",
        r"\caption{MQAR cap comparison, accuracy (\%), mean $\pm$ sample SD across all three seeds. Capped models share raw initialization, data, native attention temperature, unit-norm LayerNorm and training budgets; each is trained from scratch. The cap applies to all Transformer projections, with an uncapped vocabulary readout. The ordinary Transformer uses trainable LayerNorm and uncapped weights. Longer contexts also contain more memory pairs. All complete cohorts are retained; the cap comparisons are exploratory.}",
        r"\label{tab:mqar-cap-series}",r"\end{table}"]
    (args.output/"cap_series_table.tex").write_text("\n".join(latex)+"\n")
    artifacts={}
    for name in ("summary.json","summary.txt","per_seed.csv","aggregate.csv","cap_activity.csv",
                 "cap_series.pdf","cap_series.png","cap_series_table.tex"):
        path=args.output/name
        artifacts[name]={"sha256":file_hash(path),"bytes":path.stat().st_size}
    artifacts[str(Path(__file__).relative_to(ROOT))]={"sha256":file_hash(Path(__file__)),"bytes":Path(__file__).stat().st_size}
    save_json(args.output/"report_manifest.json",{"scope":"Local MQAR cap-series report; no remote publication","files":artifacts})
    print("\n".join(lines))


if __name__=="__main__":
    main()
