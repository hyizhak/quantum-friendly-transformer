"""Summarize the classical native-temperature cap comparison."""

import argparse
import csv
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]


def read_csv(path):
    with path.open(newline="") as stream: return list(csv.DictReader(stream))


def save_csv(path,rows):
    with path.open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(root):
    output=root/"native_cap_comparison"
    output.mkdir(exist_ok=True)
    audit=json.loads((output/"audit.json").read_text())
    assert audit["all_checks_passed"]
    per_seed=[]
    for cap in (24,64):
        for train_n in (1024,2048):
            directory=root/f"long_recall_native{cap}_{train_n}"
            plan=json.loads((directory/"plan.json").read_text())
            assert all(r["status"]=="completed" for r in plan)
            for item in plan:
                run=directory/f"{item['variant']}_seed{item['seed']}"
                result=json.loads((run/"result.json").read_text())
                metrics=read_csv(run/"test.csv")
                weights=read_csv(run/"weights_evaluation_float64.csv")
                for evaluation in result["evaluation"]:
                    n=evaluation["N"]
                    batches=[r for r in metrics if int(r["N"])==n]
                    count=sum(int(r["examples"]) for r in batches)
                    mean=lambda key:sum(float(r[key])*int(r["examples"]) for r in batches)/count
                    per_seed.append({"cap":cap,"train_N":train_n,"variant":item["variant"],"seed":item["seed"],
                        **evaluation,"matching_probability":mean("mean_target_probability"),
                        "mean_output_gap":mean("mean_pair_hidden_gap"),"mean_margin":mean("mean_margin"),
                        "minimum_sigma1":min(float(r["minimum_sigma1"]) for r in batches),
                        "minimum_sigma2":min(float(r["minimum_sigma2"]) for r in batches),
                        "active_caps":sum(r["cap_active"]=="True" for r in weights),
                        "maximum_effective_weight_norm":max(float(r["effective_frobenius"]) for r in weights),
                        "q_norm":float(next(r for r in weights if r["matrix"]=="q")["effective_frobenius"]),
                        "k_norm":float(next(r for r in weights if r["matrix"]=="k")["effective_frobenius"]),
                        "checkpoint_sha256":result["checkpoint_sha256"]})
    groups={}
    for row in per_seed: groups.setdefault((row["cap"],row["train_N"],row["variant"],row["N"]),[]).append(row)
    aggregates=[]
    measures=["accuracy","uniform_accuracy","shuffled_query_accuracy","noise_0.0001_accuracy","noise_0.01_accuracy","noise_0.1_accuracy","matching_probability","mean_output_gap","mean_margin"]
    for (cap,train_n,variant,n),rows in sorted(groups.items()):
        assert len(rows)==3
        item={"cap":cap,"train_N":train_n,"variant":variant,"N":n,"seeds":3}
        for key in measures:
            values=[r[key] for r in rows]
            item[key+"_mean"]=statistics.mean(values)
            item[key+"_sd"]=statistics.stdev(values)
        item.update(minimum_sigma1=min(r["minimum_sigma1"] for r in rows),minimum_sigma2=min(r["minimum_sigma2"] for r in rows),
                    maximum_effective_weight_norm=max(r["maximum_effective_weight_norm"] for r in rows))
        aggregates.append(item)
    save_csv(output/"per_seed.csv",per_seed)
    save_csv(output/"aggregate.csv",aggregates)
    (output/"summary.json").write_text(json.dumps({"new_training_runs":18,"existing_checkpoints_reevaluated":18,
        "shared_test_seed":1279933,"auxiliary_matching_index_supervision":True,"aggregates":aggregates,"audit":audit},indent=2)+"\n")
    lookup={(r["cap"],r["train_N"],r["variant"],r["N"]):r for r in aggregates}
    lines=["NATIVE ATTENTION TEMPERATURE, CAP24 VS CAP64","",
        "Classical training/evaluation only. Both caps use the same task, initialization, training batches, optimizer, budget and fresh test set.",
        "18 cap64 runs are new; 18 cap24 checkpoints are reevaluated without retraining.",
        "All three seeds11,22,33 are retained. Reported SD is sample SD across training seeds.",
        "All cohorts use additional matching-index supervision during training only; this information never enters inference.",
        "Dimension32. Curriculum500 updates at N64,500 at N256,2000 at the stated maximum. Fixed final checkpoints.",
        "Evaluation2048 independent pairs/4096 sequences per length, fresh test seed1279933.",""]
    for row in aggregates:
        if row["N"] not in (row["train_N"],8192): continue
        lines.append(f"cap={row['cap']:2d} trainN={row['train_N']} {row['variant']:17s} testN={row['N']}: "
            f"{100*row['accuracy_mean']:.2f} +/- {100*row['accuracy_sd']:.2f}%; "
            f"uniform ablation={100*row['uniform_accuracy_mean']:.2f}%, query shuffle={100*row['shuffled_query_accuracy_mean']:.2f}%, "
            f"noise L2 .1={100*row['noise_0.1_accuracy_mean']:.2f}%")
    lines += ["","Interpretation:",
        "Native temperature remains sqrt(32); cap64 is a larger bound and is not an FN24 recovery.",
        "The pairs have identical mean-pooled inputs/opposite answers, giving this uniform-attention architecture exactly50%.",
        "Only the strict variants use alpha0=N*cap^2. Native successes must not be attributed to that strict scaling.",
        "The follow-up is exploratory and was motivated by the earlier cap24 results; a fresh test set does not erase that history.",
        "All original classification-only and auxiliary results remain available under long_recall_summary."]
    (output/"summary.txt").write_text("\n".join(lines)+"\n")
    styles=[(24,"native_sqrt_d","Native temperature, cap24","#e18b2a","-"),
            (64,"native_sqrt_d","Native temperature, cap64","#1764ab","-"),
            (24,"strict_frobenius","Strict scaling, cap24","#ba5640",":"),
            (64,"strict_frobenius","Strict scaling, cap64","#754ba2","-."),
            (24,"uniform","Mean pooling, cap24","#555555","--"),
            (64,"uniform","Mean pooling, cap64","#888888",":")]
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.5),sharey=True,constrained_layout=True)
    for ax,train_n in zip(axes,(1024,2048)):
        for cap,variant,label,color,style in styles:
            rows=sorted([r for r in aggregates if (r["cap"],r["train_N"],r["variant"])==(cap,train_n,variant)],key=lambda r:r["N"])
            x=[r["N"] for r in rows]; y=[100*r["accuracy_mean"] for r in rows]; sd=[100*r["accuracy_sd"] for r in rows]
            ax.plot(x,y,marker="o",markersize=3,linestyle=style,color=color,label=label)
            ax.fill_between(x,[max(0,a-b) for a,b in zip(y,sd)],[min(100,a+b) for a,b in zip(y,sd)],color=color,alpha=.1)
        ax.axvline(train_n,color="#bbbbbb",linestyle=":")
        ax.set_xscale("log",base=2)
        ax.set_xticks([256,512,1024,2048,4096,8192],["256","512","1024","2048","4096","8192"])
        ax.set(title=f"Curriculum through N={train_n}",xlabel="Evaluation sequence length N",ylim=(47,102))
        ax.grid(alpha=.15)
    axes[0].set_ylabel("Accuracy (%)")
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="outside lower center",ncol=3,fontsize=8)
    fig.suptitle("Classical recall with matching-index supervision; all three fixed seeds",fontsize=11)
    for ext in ("pdf","png"): fig.savefig(output/f"native_caps.{ext}",dpi=180)
    plt.close(fig)
    table=[r"\begin{table}[t]",r"\centering\small",r"\begin{tabular}{llrrrr}",r"\hline",
        r"Cap & Attention & \multicolumn{2}{c}{Train through $N=1024$} & \multicolumn{2}{c}{Train through $N=2048$} \\",
        r" & & $N=1024$ & $N=8192$ & $N=2048$ & $N=8192$ \\",r"\hline"]
    for cap in (24,64):
        for variant,label in [("native_sqrt_d","Native"),("strict_frobenius","Strict"),("uniform","Uniform")]:
            values=[]
            for t,n in ((1024,1024),(1024,8192),(2048,2048),(2048,8192)):
                r=lookup[cap,t,variant,n]; values.append(f"${100*r['accuracy_mean']:.2f}\\pm{100*r['accuracy_sd']:.2f}$")
            table.append(f"{cap} & {label} & "+" & ".join(values)+r" \\")
    table += [r"\hline",r"\end{tabular}",r"\caption{Classical paired recall accuracy (\%), mean $\pm$ sample SD over all three seeds. Native scaling remains $\sqrt{32}$; strict scaling is $N$ times the squared cap. Every setting uses the same disclosed matching-index auxiliary training objective and budget. Cap24 checkpoints are reevaluated, not retrained, on the fresh test set shared with cap64. Cap64 is a larger norm allowance and does not constitute FN24 recovery.}",r"\label{tab:native-cap-recall}",r"\end{table}"]
    (output/"native_caps_table.tex").write_text("\n".join(table)+"\n")
    print("\n".join(lines))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results",type=Path,required=True)
    summarize(parser.parse_args().results)
