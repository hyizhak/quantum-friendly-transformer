"""Compare cap48 with cap64 and the matched ordinary Transformer on MQAR."""

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

from experiments.lib.mqar_reporting import cohort_config, protocol_summary
from experiments.lib.experiment import save_csv,save_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--standard-results", type=Path, required=True)
    parser.add_argument("--reference-results", type=Path, required=True)
    parser.add_argument("--capped-results", type=Path, required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    models=[("standard","Standard Transformer",args.standard_results,"native"),
            ("cap64","QFT adaptation (cap 64)",args.reference_results,"native"),
            ("cap48","QFT adaptation (cap 48)",args.capped_results,"native"),
            ("uniform64","Uniform control (cap 64)",args.reference_results,"uniform"),
            ("uniform48","Uniform control (cap 48)",args.capped_results,"uniform")]
    records,norms,inputs=[],[],{}
    for model,title,root,variant in models:
        audit_path=root/"audit.json"
        audit=json.loads(audit_path.read_text())
        if audit["status"]!="passed":
            raise AssertionError("Each cohort must pass its audit")
        inputs[str(audit_path.resolve())]=hashlib.sha256(audit_path.read_bytes()).hexdigest()
        for seed in (11,22,33):
            folder=root/f"{variant}_seed{seed}"
            path=folder/"result.json"
            result=json.loads(path.read_text())
            if result["smoke"]:
                raise AssertionError("Smoke result in benchmark summary")
            inputs[str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest()
            for row in result["evaluation"]:
                records.append({"model":model,"label":title,"seed":seed,
                    **{k:v for k,v in row.items() if k!="layer_diagnostics"}})
            for stage in ("initial","final"):
                with (folder/f"weights_{stage}.csv").open() as stream:
                    weights=list(csv.DictReader(stream))
                selected=[r for r in weights if r["capped"]=="True"]
                norms.append({"model":model,"seed":seed,"stage":stage,"capped_matrices":len(selected),
                    "active_caps":sum(r["cap_active"]=="True" for r in selected),
                    "max_effective_capped_norm":max((float(r["effective_frobenius"]) for r in selected),default=None),
                    "max_learned_gamma":max((float(r["gamma"]) for r in selected),default=None),
                    "active_matrix_names":",".join(r["matrix"] for r in selected if r["cap_active"]=="True")})
    conditions=[(64,4),(128,8),(256,16),(256,64),(512,32),(1024,64),(2048,128)]
    aggregates=[]
    for model,title,_,_ in models:
        for n,k in conditions:
            rows=[r for r in records if (r["model"],r["N"],r["pairs"])==(model,n,k)]
            if len(rows)!=3 or {r["seed"] for r in rows}!={11,22,33}:
                raise AssertionError("Incomplete seed cohort")
            row={"model":model,"label":title,"N":n,"pairs":k,"role":rows[0]["role"],"seeds":3}
            for metric in ("accuracy","mean_nll","sequence_accuracy","uniform_ablation_accuracy"):
                row[metric+"_mean"]=statistics.mean(r[metric] for r in rows)
                row[metric+"_sd"]=statistics.stdev(r[metric] for r in rows)
            aggregates.append(row)
    save_csv(args.output/"per_seed.csv",records)
    save_csv(args.output/"aggregate.csv",aggregates)
    save_csv(args.output/"cap_activity.csv",norms)
    save_json(args.output/"summary.json",{"new_models":6,"total_comparison_models":15,
        "all_seeds_retained":True,"cap_change":[64,48],"results":aggregates,"cap_activity":norms,
        "source_result_sha256":inputs,"scope":"Exploratory cap sensitivity for the MQAR adaptation; same official data and training budget."})
    report_config = cohort_config(args.reference_results)
    summary_path = args.output / "summary.json"
    summary_record = json.loads(summary_path.read_text())
    summary_record.update(configuration=report_config,
                          workflow_validation=bool(report_config.get("workflow_validation")))
    save_json(summary_path, summary_record)
    lines=["MQAR: CAP 64 TO CAP 48", "",
        "Cap48 retrained from scratch, with the same raw initial parameters, official MQAR data,",
        protocol_summary(report_config),
        "Three seeds (11,22,33); fixed final checkpoints; answer labels only.",
        "All 12 Transformer projection matrices are capped; the vocabulary readout is uncapped.",
        "Effective initial weights can differ because the smaller cap is enforced from the start.",
        "Numbers are mean +/- sample SD across all three seeds. All prior cohorts are preserved.", ""]
    for n,k in conditions:
        lines.append(f"N={n}, memory pairs={k}")
        for r in [r for r in aggregates if (r["N"],r["pairs"])==(n,k)]:
            lines.append(f"  {r['label']}: {100*r['accuracy_mean']:.3f} +/- {100*r['accuracy_sd']:.3f}%")
        lines.append("")
    lines.append("Final cap activity:")
    for r in [r for r in norms if r["stage"]=="final" and r["capped_matrices"]]:
        lines.append(f"  {r['model']} seed{r['seed']}: {r['active_caps']}/{r['capped_matrices']} active; max effective norm {r['max_effective_capped_norm']:.4f}; active matrices {r['active_matrix_names']}")
    lines += ["", "Interpretation:",
        "This is a matched exploratory cap comparison prompted by the earlier results. Reusing the test",
        "stream makes the comparison paired; it does not make it a newly untouched confirmatory test.",
        "Longer contexts also increase memory load. The model remains the MQAR adaptation, not the",
        "legacy model class. This is not FN24 recovery or a demonstration of computational advantage.", ""]
    (args.output/"summary.txt").write_text("\n".join(lines))
    fig,ax=plt.subplots(figsize=(10,5),layout="constrained")
    for model,title,color in [("standard","Standard Transformer","#426b45"),("cap64","QFT adaptation: cap 64","#245c9f"),("cap48","QFT adaptation: cap 48","#9c3d70"),("uniform48","Uniform control: cap 48","#b05a32")]:
        rows=[next(r for r in aggregates if (r["model"],r["N"],r["pairs"])==(model,n,k)) for n,k in conditions]
        ax.plot(range(len(conditions)),[100*r["accuracy_mean"] for r in rows],marker="o",label=title,color=color)
        for index,seed in enumerate((11,22,33)):
            values=[100*next(r["accuracy"] for r in records if (r["model"],r["seed"],r["N"],r["pairs"])==(model,seed,n,k)) for n,k in conditions]
            ax.scatter(np.arange(len(conditions))+(index-1)*.055,values,s=15,color=color,alpha=.65)
    ax.set_xticks(range(len(conditions)),[f"{n}\n{k} pairs" for n,k in conditions])
    ax.set_xlabel("Sequence length and memory pairs")
    ax.set_ylabel("Official MQAR query-token accuracy (%)")
    ax.set_ylim(-2,102)
    ax.grid(axis="y",alpha=.2)
    ax.set_title("MQAR cap comparison: means and all three seeds")
    ax.legend()
    fig.savefig(args.output/"cap_comparison.pdf")
    fig.savefig(args.output/"cap_comparison.png",dpi=180)
    plt.close(fig)
    latex=[r"\begin{table}[t]",r"\centering\small",r"\begin{tabular}{lrrrr}",r"\hline",
        r"Model & $N=256,K=16$ & $N=512,K=32$ & $N=1024,K=64$ & $N=2048,K=128$ \\",r"\hline"]
    for model,title,_,_ in models:
        cells=[]
        for n,k in [(256,16),(512,32),(1024,64),(2048,128)]:
            r=next(r for r in aggregates if (r["model"],r["N"],r["pairs"])==(model,n,k))
            cells.append(f"${100*r['accuracy_mean']:.3f}\\pm{100*r['accuracy_sd']:.3f}$")
        latex.append(title+" & "+" & ".join(cells)+r" \\")
    latex += [r"\hline",r"\end{tabular}",
        r"\caption{Matched MQAR cap comparison, query-token accuracy (\%), mean $\pm$ sample SD over all three seeds. Cap48 and cap64 use identical raw initialization, training data, native attention temperature, normalization and update budgets. Each cap is enforced during training and evaluation on all Transformer projections; the vocabulary readout is uncapped. Both complete cohorts and their uniform controls are retained. The comparison is exploratory.}",
        r"\label{tab:mqar-cap-comparison}",r"\end{table}"]
    (args.output/"cap_comparison_table.tex").write_text("\n".join(latex)+"\n")
    print("\n".join(lines))


if __name__=="__main__":
    main()
