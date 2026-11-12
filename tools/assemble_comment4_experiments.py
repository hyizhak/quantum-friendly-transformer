"""Update only the Comment 4 response body using classical experiment results."""

import csv
import hashlib
import json
from pathlib import Path
import re
import statistics

ROOT=Path(__file__).resolve().parents[1]


def percentage(row):
    return f"${100*row['accuracy_mean']:.2f}\\pm{100*row['accuracy_sd']:.2f}\\%$"


def mqar_primary_evidence():
    """Summarize existing predictions and norms; perform no new inference."""
    summary=json.loads((ROOT/"results/reviewer/mqar_cap24_comparison/summary.json").read_text())
    aggregates={(r["model"],r["N"],r["pairs"]):r for r in summary["results"]}
    config=json.loads((ROOT/"experiments/configs/reviewer/mqar_cap24.json").read_text())
    d,n,pairs=config["dimension"],config["train_length"],config["train_pairs"]
    sources={}
    predictions={}
    common_queries=None
    for model,cohort,variant in (("standard","mqar_standard","native"),
                                 ("cap24","mqar_cap24","native"),
                                 ("uniform24","mqar_cap24","uniform")):
        predictions[model]=[]
        for seed in config["seeds"]:
            run=ROOT/f"results/reviewer/{cohort}/{variant}_seed{seed}"
            result_path=run/"result.json"
            expected=summary["source_result_sha256"][str(result_path.relative_to(ROOT))]
            assert hashlib.sha256(result_path.read_bytes()).hexdigest()==expected
            result=json.loads(result_path.read_text())
            metric=next(r for r in result["evaluation"] if (r["N"],r["pairs"])==(n,pairs))
            path=run/f"test_N{n}_K{pairs}_predictions.csv"
            with path.open(newline="") as stream:
                records=list(csv.DictReader(stream))
            queries=[(int(r["sequence"]),int(r["position"]),int(r["label"])) for r in records]
            if common_queries is None:
                common_queries=queries
            assert queries==common_queries, "Models must use identical test queries"
            assert len(records)==metric["queries"]==config["test_examples"]*pairs
            selected=[r for r in records if int(r["position"])+1>=d]
            measured={"seed":seed,"all_queries":len(records),"prefix_ge_d_queries":len(selected)}
            for field in ("prediction","uniform_prediction"):
                key="accuracy" if field=="prediction" else "uniform_ablation_accuracy"
                measured[key]=sum(r[field]==r["label"] for r in records)/len(records)
                assert abs(measured[key]-metric[key])<1e-12, (path,key)
                measured["prefix_ge_d_"+key]=sum(r[field]==r["label"] for r in selected)/len(selected)
            predictions[model].append(measured)
            for source in (path,result_path):
                sources[str(source.relative_to(ROOT))]=hashlib.sha256(source.read_bytes()).hexdigest()

    labels={"standard":"Ordinary Transformer", "cap24":"Our MQAR model, cap 24",
            "uniform24":"Uniform attention, trained (cap 24)",
            "ablation24":"Cap-24 checkpoint, uniform substitution"}
    rows={}
    for model in labels:
        records=predictions["cap24" if model=="ablation24" else model]
        metric="uniform_ablation_accuracy" if model=="ablation24" else "accuracy"
        row={"label":labels[model],"all_queries":records[0]["all_queries"],
             "prefix_ge_d_queries":records[0]["prefix_ge_d_queries"],
             "per_seed":[{"seed":r["seed"],"accuracy":r[metric],
                          "prefix_ge_d_accuracy":r["prefix_ge_d_"+metric]} for r in records]}
        for prefix,field in (("",metric),("prefix_ge_d_","prefix_ge_d_"+metric)):
            row[prefix+"accuracy_mean"]=statistics.mean(r[field] for r in records)
            row[prefix+"accuracy_sd"]=statistics.stdev(r[field] for r in records)
        aggregate=aggregates["cap24" if model=="ablation24" else model,n,pairs]
        for moment in ("mean","sd"):
            assert abs(row["accuracy_"+moment]-aggregate[metric+"_"+moment])<1e-12
        rows[model]=row

    def cell(row,prefix=""):
        return f"${100*row[prefix+'accuracy_mean']:.3f}\\pm{100*row[prefix+'accuracy_sd']:.3f}$"
    table=[r"\begin{table}[tbp]",r"\centering\small",r"\begin{tabular}{lrr}",r"\hline",
           r"Model / attention setting & All queries & Prefix length $\geq d$ \\",r"\hline"]
    for row in rows.values():
        table.append(row["label"]+" & "+cell(row)+" & "+cell(row,"prefix_ge_d_")+r" \\")
    table.extend([r"\hline",r"\end{tabular}",
        rf"\caption{{MQAR query-token accuracy (\%) at $N={n}$, $d={d}$, and $K={pairs}$ memory pairs: mean $\pm$ sample standard deviation across all three training seeds, evaluated at fixed final checkpoints on the same {config['test_examples']}-sequence test set. The primary column scores all {rows['cap24']['all_queries']} labeled queries. The second column reaggregates the same saved predictions for the {rows['cap24']['prefix_ge_d_queries']} queries whose causal prefix has at least $d$ tokens. Uniform substitution replaces attention in both blocks of each trained cap-24 checkpoint, with all other weights fixed and no retraining.}}",
        r"\label{tab:comment4-mqar-primary}",r"\end{table}",""])
    seed_table=[r"\begin{table}[tbp]",r"\centering\small",r"\begin{tabular}{lrrr}",r"\hline",
                r"Model / attention setting & Seed 11 & Seed 22 & Seed 33 \\",r"\hline"]
    for row in rows.values():
        seed_table.append(row["label"]+" & "+" & ".join(
            f"{100*r['accuracy']:.3f}" for r in row["per_seed"])+r" \\")
    seed_table.extend([r"\hline",r"\end{tabular}",
        r"\caption{Individual final-checkpoint scores (\%) for all labeled queries in the primary MQAR condition. These are all three training seeds; none is selected or omitted.}",
        r"\label{tab:comment4-mqar-seeds}",r"\end{table}",""])

    norms={}
    for stage in ("initial","final"):
        norms[stage]=[]
        for seed in config["seeds"]:
            path=ROOT/f"results/reviewer/mqar_cap24/native_seed{seed}/weights_{stage}.csv"
            with path.open(newline="") as stream:
                norms[stage].extend({**r,"seed":seed} for r in csv.DictReader(stream) if r["capped"]=="True")
            sources[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
        assert len(norms[stage])==36
        assert sum(r["cap_active"]=="True" for r in norms[stage])==18
    norm_table=[r"\begin{table}[tbp]",r"\centering\footnotesize",r"\begin{tabular}{lrrrcc}",r"\hline",
                r"Matrix & Initial raw & Final raw & Final effective & Active: initial & final \\",r"\hline"]
    norm_groups={}
    for kind,label in (("q",r"$W_q$"),("k",r"$W_k$"),("v",r"$W_v$"),
                       ("o",r"$W_o$"),("m1",r"$M_1$"),("m2",r"$M_2$")):
        norm_groups[kind]={}
        cells=[]
        for stage,field in (("initial","raw_frobenius"),("final","raw_frobenius"),("final","effective_frobenius")):
            values=[float(r[field]) for r in norms[stage] if r["matrix"].endswith("."+kind)]
            assert len(values)==6
            norm_groups[kind][stage+"_"+field]=[min(values),max(values)]
            cells.append(f"{min(values):.3f}--{max(values):.3f}")
        for stage in ("initial","final"):
            active=sum(r["cap_active"]=="True" for r in norms[stage] if r["matrix"].endswith("."+kind))
            norm_groups[kind][stage+"_active"]=active
            cells.append(f"{active}/6")
        norm_table.append(label+" & "+" & ".join(cells)+r" \\")
    norm_table.extend([r"\hline",r"\end{tabular}",
        r"\caption{Frobenius norms for the cap-24 MQAR models. Each range pools the two blocks and three seeds (six matrices of each type). Initial raw norms are measured after the disclosed initialization adjustment, before applying the forward-pass cap. Activity means that the raw norm exceeds the learned threshold $\gamma$, whose maximum is 24. The embedding and vocabulary readout matrices are outside this projection cap.}",
        r"\label{tab:comment4-mqar-norms}",r"\end{table}",""])
    prefix_lengths=[position+1 for _,position,_ in common_queries]
    diagnostics={"N":n,"d":d,"pairs":pairs,"rows":rows,"norm_groups":norm_groups,
                 "causal_prefix_length":{"min":min(prefix_lengths),"median":statistics.median(prefix_lengths),
                                         "max":max(prefix_lengths)},
                 "source_sha256":sources,
                 "scope":"Descriptive reaggregation of existing predictions and weight records; no new training or inference. The prefix threshold is the model width d. This is not a new confirmatory test."}
    return "\n".join(table),"\n".join(seed_table),"\n".join(norm_table),diagnostics


def mqar_evidence():
    """Render the audited MQAR results without changing frozen run artifacts."""
    summary_path=ROOT/"results/reviewer/mqar_cap24_comparison/summary.json"
    summary=json.loads(summary_path.read_text())
    for name, expected in summary["source_result_sha256"].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected, name
    for cohort in ("mqar", "mqar_variance_init", "mqar_cap48", "mqar_cap24", "mqar_standard"):
        audit=json.loads((ROOT/"results/reviewer"/cohort/"audit.json").read_text())
        assert audit["status"]=="passed", cohort
    rows={(r["model"],r["N"],r["pairs"]):r for r in summary["results"]}
    assert all(r["seeds"]==3 for r in rows.values())
    def score(model, n=256, pairs=16, unit=True):
        row=rows[model,n,pairs]
        return (f"${100*row['accuracy_mean']:.3f}\\pm{100*row['accuracy_sd']:.3f}"
                +(r"\%$" if unit else "$"))

    config_path=ROOT/"experiments/configs/reviewer/mqar_cap24.json"
    config=json.loads(config_path.read_text())
    assert (config["cap"],config["train_length"],config["train_pairs"],
            config["train_examples"],config["epochs"],config["seeds"])==(
                24,256,16,20000,32,[11,22,33])
    final=[r for r in summary["cap_activity"] if r["model"]=="cap24" and r["stage"]=="final"]
    assert len(final)==3 and all(r["active_caps"]==6 and r["capped_matrices"]==12 for r in final)
    max_effective=max(r["max_effective_capped_norm"] for r in final)
    initial_active=[]
    for seed in config["seeds"]:
        path=ROOT/f"results/reviewer/mqar_cap24/native_seed{seed}/weights_initial.csv"
        with path.open(newline="") as stream:
            initial_active.extend(float(r["raw_frobenius"]) for r in csv.DictReader(stream)
                                  if r["cap_active"]=="True")
    assert len(initial_active)==18
    initial_summary_path=ROOT/"results/reviewer/mqar_summary/summary.json"
    original=json.loads(initial_summary_path.read_text())
    initial=next(r for r in original["aggregates"] if
                 (r["cohort"],r["variant"],r["N"],r["pairs"])==("initial","native",256,16))
    initial_score=f"${100*initial['accuracy_mean']:.3f}\\pm{100*initial['accuracy_sd']:.3f}\\%$"
    drop=100*(rows["cap48",256,16]["accuracy_mean"]-rows["cap24",256,16]["accuracy_mean"])

    table=[r"\begin{table}[tbp]",r"\centering",r"\begingroup",
           r"\footnotesize\setlength{\tabcolsep}{3pt}",r"\begin{tabular}{rrrrrr}",r"\hline",
           r"$N$ & $K$ & Standard & QFT cap 64 & QFT cap 48 & QFT cap 24 \\",r"\hline"]
    for condition in config["test_conditions"]:
        n,pairs=condition["N"],condition["pairs"]
        table.append(f"{n} & {pairs} & "+" & ".join(
            score(model,n,pairs,unit=False) for model in ("standard","cap64","cap48","cap24"))+r" \\")
    table.extend([r"\hline",r"\end{tabular}",r"\endgroup",
        r"\caption{Controlled MQAR evaluation using the official generator: query-token accuracy (\%), mean $\pm$ sample standard deviation over three seeds. All models are trained at $N=256$ with $K=16$ key--value pairs. Other rows evaluate the same final checkpoints without retraining; both length and memory load vary. Standard denotes the locally trained ordinary Transformer. QFT denotes the MQAR adaptation with the stated projection cap and an uncapped vocabulary readout. All cap settings share the QFT recipe, raw initial parameters, data, and update budget.}",
        r"\label{tab:comment4-mqar-caps}",r"\end{table}",""])
    text=rf"""\textbf{{MQAR: an established retrieval task with a disclosed training protocol.}}
We evaluate Multi-Query Associative Recall (MQAR) to measure content-dependent
retrieval with constrained projection norms. The task was introduced in
\href{{https://arxiv.org/abs/2312.04927}}{{\emph{{Zoology: Measuring and Improving Recall in Efficient Language Models}}}}.
We use the unchanged
\href{{https://github.com/HazyResearch/zoology/blob/{config['upstream_commit']}/zoology/data/multiquery_ar.py}}{{official generator at a pinned revision}},
with vocabulary size 8192, random non-query tokens, power-law parameter
$0.01$, and one presentation of the memory pairs. We score every labeled
query over the full vocabulary. The controlled training and evaluation
protocol is specified below.

The QFT adaptation has two causal Transformer blocks, one attention head,
$d=64$, a feed-forward width of 128, GELU activations, and fixed sinusoidal
positions. It uses native attention scaling $\alpha_0=\sqrt{{64}}=8$ in both
training and evaluation. Input rows are normalized, and both LayerNorms in
each block have fixed gain $1/\sqrt d$, zero bias, and zero epsilon.
All 12 complete projections---$W_q,W_k,W_v,W_o,M_1,M_2$ in each block---use
the forward-pass parametrization
$W_{{\rm eff}}=\min(1,\gamma/(\|W_{{\rm raw}}\|_F+10^{{-12}}))W_{{\rm raw}}$,
where $\gamma=C\operatorname{{sigmoid}}(g)$ and $C$ is 24, 48, or 64.
The embedding matrix and the vocabulary readout are not Frobenius-capped;
the embedding outputs are row-normalized. All trainable parameters are
optimized from scratch. These are MQAR adaptations of our normalization
and capping components, separate from the pretrained SST-2 models.

Every model trains on 20000 sequences at $N=256$ with 16 key--value pairs,
with separate 1000-sequence validation and test sets; each additional test
condition also has 1000 sequences. We retain seeds 11, 22, and 33 and train
for 32 epochs (5024 updates), using batch size 128 and AdamW with learning
rate $0.003$, weight decay $0.01$, 5\% warmup, cosine decay, and gradient
clipping at one. Supervision consists only of the answer-token labels;
no matching-token indices are supplied. Evaluation uses the fixed final
checkpoint for every seed. Within each cap setting, separately trained
causal uniform-attention controls have paired initial parameters and
minibatch orders. This control tests the contribution of content-dependent
attention within the specified architecture and training budget.

Our ordinary Transformer comparator has the same width, depth, positional
features, data, and training budget, with uncapped weights, ordinary
trainable LayerNorm (initial gain one, zero bias, epsilon $10^{{-5}}$), and
input features scaled by $\sqrt d$. We train this comparator locally. At the
primary condition, where $N/d=4$, the cap-24 adaptation achieves
{score('cap24')}, compared with {score('standard')} for the ordinary
Transformer and {score('uniform24')} for the matched cap-24 uniform-attention
control. The cap-64 and cap-48 adaptations achieve {score('cap64')} and
{score('cap48')}; their respective uniform controls achieve
{score('uniform64')} and {score('uniform48')}. All uncertainties are sample
standard deviations across the three training seeds. Reducing the cap from
48 to 24 costs {drop:.3f} percentage points on this condition. These results
support useful content-dependent retrieval under this protocol.
The comparison with the ordinary Transformer includes normalization and
initialization differences; the cap-only comparison uses the same QFT
recipe. Table~\ref{{tab:comment4-mqar-primary}} also reports substitution
of uniform attention in the trained cap-24 checkpoints. This intervention
changes both attention blocks while keeping all other trained parameters
fixed, without further optimization. The causal-prefix column reaggregates
saved predictions at positions $t$ with $t+1\geq d$ (zero-based indexing).
The primary metric includes all labeled queries.
Table~\ref{{tab:comment4-mqar-seeds}} gives each
training seed's primary score.

\input{{results/reviewer/comment4_update/comment4_mqar_primary_table.tex}}
\input{{results/reviewer/comment4_update/comment4_mqar_seed_table.tex}}

The cap-24 constraint is active on $Q$, $K$, and $M_1$ in both blocks:
six of twelve projections in every seed at initialization and at the final
checkpoint. The initially active matrices have raw Frobenius norms from
{min(initial_active):.3f} to {max(initial_active):.3f}, above the maximum cap
of 24. Across all final cap-24 models, the largest effective projection norm
is {max_effective:.3f}. Initial and final raw/effective norms and the learned
cap thresholds are recorded for every matrix. Table~\ref{{tab:comment4-mqar-norms}}
summarizes the norms before and after capping.

\input{{results/reviewer/comment4_update/comment4_mqar_norm_table.tex}}

The initial cap-64 cohort with default initialization scored
{initial_score} on the primary condition. Training and validation
diagnostics motivated an initialization adjustment for unit-norm
activations. The follow-up used a fresh test stream fixed before training,
and the original cohort is retained. The capped models in
Table~\ref{{tab:comment4-mqar-caps}} share this adjusted initialization method,
raw initial parameters, data, and training budget, although their effective
initial weights differ where clipping is active. Specifically, at
initialization only, $W_q,W_k,M_1$ are multiplied by $\sqrt d$, $M_2$ is
divided by $\sqrt d$, and the vocabulary readout is multiplied by $\sqrt d$,
relative to the original random draws. The forward-pass caps and native
attention denominator remain in force. Subsequent cap comparisons and
the ordinary Transformer baseline were added after the earlier results,
using the same follow-up test stream. Each cohort's protocol was fixed
before its training. All seeds and earlier cohorts are retained.

Table~\ref{{tab:comment4-mqar-caps}} reports all seven attention-model test
conditions. At $N=2048$ with 128 key--value pairs, the cap-24 model scores
{score('cap24',2048,128)}, while the ordinary Transformer scores
{score('standard',2048,128)}. These rows evaluate extrapolation from training
at $N=256$ while increasing both sequence length and memory load. They
measure the joint effect of these changes on the same trained checkpoints.
Code, configurations, all per-query predictions, checkpoints,
and matrix diagnostics are retained. Artifact checks cover every stored
query, and fresh inference replays cover the first evaluation batch for
every condition and model.

\input{{results/reviewer/comment4_update/comment4_mqar_table.tex}}

"""
    inputs={str(p.relative_to(ROOT))+"_sha256":hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (summary_path,config_path,initial_summary_path)}
    return text,"\n".join(table),inputs


def focused_response(evidence):
    """Lead with the experimental answer and state the methods directly."""
    rows=evidence["rows"]
    def score(model,prefix=""):
        row=rows[model]
        return f"${100*row[prefix+'accuracy_mean']:.3f}\\pm{100*row[prefix+'accuracy_sd']:.3f}\\%$"
    return rf"""We have added a controlled Multi-Query Associative Recall (MQAR)
evaluation to test content-dependent retrieval at $N\gtrsim d$.
With $N=256$, $d=64$, native attention scaling, and a maximum projection
cap of 24, our model achieves {score('cap24')} query accuracy across three
training seeds. The cap is active in every seed. A matched trained
uniform-attention control scores {score('uniform24')}, and substituting
uniform attention in the trained cap-24 checkpoints gives
{score('ablation24')}. These results show that active projection caps of
24 are compatible with effective content-dependent retrieval in this
setting. The training protocol, controls, and cap diagnostics address
the individual questions as follows.

\textbf{{(a) Attention scaling during training and evaluation.}}
The MQAR models use the fixed denominator $\alpha_0=\sqrt d=8$ throughout
training and evaluation. The projection caps and attention denominator
are specified separately. For the recorded SST-2 recovery runs, the
bounded family enforces
$\alpha_{{0,\ell}}\geq\alpha_{{s,\ell}}^2\,24^2$, and the learned denominator
can increase above this floor. Here $\alpha_s$ is a token-norm calibration
constant; complete attention-input matrix bounds were not certified for
these SST-2 runs. The supplement specifies the calibration, scaling,
and training scope for each recorded family, including the separate
unrestricted-temperature runs. The original SST-2 table is retained
as a classification result, with these recovery experiments reported
separately.

\textbf{{(b)--(c) MQAR at $N\gtrsim d$ and attention controls.}}
We use the unchanged official MQAR generator from
\href{{https://arxiv.org/abs/2312.04927}}{{\emph{{Zoology: Measuring and Improving Recall in Efficient Language Models}}}},
with vocabulary size 8192, random non-query tokens, and power-law parameter
$0.01$. Our MQAR model implements the proposed normalization and
projection-cap components in two causal blocks: one attention head,
$d=64$, feed-forward width 128, GELU, sinusoidal positions, normalized
input rows, and fixed LayerNorm gain $1/\sqrt d$ with zero bias and epsilon.
All trainable parameters are optimized from scratch.

Training and the primary evaluation use $N=256$ tokens and 16 key--value
pairs, giving $N/d=4$. We score every labeled query over the full
vocabulary. Table~\ref{{tab:comment4-mqar-primary}} reports all 16000 test
queries and also the 8168 queries whose actual causal prefix contains at
least $d$ tokens. On the latter subset of the same saved predictions,
our model achieves {score('cap24','prefix_ge_d_')}.

The two attention controls directly test whether the retrieval performance
depends on content-dependent attention. The trained uniform control uses
causal averaging in both blocks, with the same remaining architecture,
cap, paired initial parameters, data, minibatch order, and update budget.
The checkpoint substitution replaces both attention distributions with
causal uniform weights, keeping all trained parameters fixed. The accuracy reduction
in both controls shows that the trained model's retrieval performance
depends on content-dependent attention.

The ordinary Transformer reference achieves {score('standard')} with the
same width, depth, positional features, data, and update budget. It uses
uncapped projections, ordinary trainable LayerNorm, and input features
scaled by $\sqrt d$. This reference compares the complete models; the
separate cap comparison (24, 48, 64) holds the QFT recipe and raw initial
parameters fixed. The supplement reports every evaluated length and
memory load, including extrapolation from training at $N=256$.

\input{{results/reviewer/comment4_update/comment4_mqar_primary_table.tex}}

\textbf{{(d) Training seeds and checkpoint selection.}}
MQAR uses 20000 training sequences and separately generated validation
and test sets of 1000 sequences each. We report all three training seeds
(11, 22, 33), evaluated on the same test sequences after 32 epochs
(5024 updates). Every score uses the fixed final checkpoint. Supervision
contains only answer-token labels. The reported mean and sample standard
deviation summarize the three training seeds; individual scores and
complete optimizer settings are provided in the supplement.

The initialization adjustment was based on training and validation
diagnostics, with a fresh test stream fixed before the follow-up training.
Subsequent cap comparisons and the ordinary Transformer reference were
added using that same test stream; their protocols were fixed before
each cohort was trained. The supplement retains the earlier cohorts and
documents this sequence of experiments.

For SST-2, the completed fixed cohorts use an internal training split for
checkpoint selection and report seed variability on the official
validation set. The lower-rate TinyStories-8M cohort scores
$83.30\pm0.46\%$, and its uniform control scores $82.87\pm0.89\%$.
These results quantify classification performance, while the MQAR controls
provide the direct test of content-dependent retrieval. The supplement
reports fixed-cohort statistics and adaptive recovery searches separately,
with the selection rules and all attempted runs recorded.

\textbf{{(e) Capped matrices, enforcement, and measured activity.}}
In MQAR, all 12 complete projections ($W_q,W_k,W_v,W_o,M_1,M_2$ in each
block) use
\[
 W_{{\mathrm{{eff}}}}=\min\!\left(1,
 \frac{{\gamma}}{{\lVert W_{{\mathrm{{raw}}}}\rVert_F+10^{{-12}}}}\right)
 W_{{\mathrm{{raw}}}},\qquad \gamma=C\,\operatorname{{sigmoid}}(g).
\]
The learned threshold $\gamma$ is bounded by $C=24$, and capping is
enforced on every forward pass. In every seed, $W_q,W_k,M_1$ in both
blocks are actively capped at initialization and at the final checkpoint
(6/12 matrices). Their initial raw norms range from 36.367 to 52.547;
the largest final effective projection norm is 23.991. The embedding
and vocabulary readout matrices are uncapped, and embedding outputs are
row-normalized. Initial raw and final raw/effective norms are tabulated
for every projection type in the supplement.

For SST-2, the cap covers attention query/key/value/output and feed-forward
projections, including gate/up/down matrices, and excludes embeddings and
the classification head. TinyStories and OLMo have full-parameter runs.
The Mistral records distinguish frozen-base updates to the head,
normalization, cap, and scale parameters from later low-rank and
full-parameter runs. Its initial attention-matrix norms are 4.08--17.76,
below 24. The supplement reports the activity of each capped projection
and the adaptation scope of each run.

"""


def training8192_evidence():
    path=ROOT/"results/reviewer/long_recall_train8192_summary/summary.json"
    if not path.exists():
        return "","",{}
    data=json.loads(path.read_text())
    assert data["validation_audit"]["status"]==data["final_test_audit"]["status"]=="passed"
    for name,expected in data["source_result_sha256"].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,name
    rows={(r["group"],r["N"]):r for r in data["aggregates"]}
    def score(group):
        r=rows[group,8192]
        return f"${100*r['accuracy_mean']:.3f}\\pm{100*r['accuracy_sd']:.3f}\\%$"
    selection=data["selection"]
    chosen=next(r for r in selection["recipes"] if r["recipe"]==selection["selected_recipe"])
    config=json.loads((ROOT/chosen["root"]/"config.json").read_text())
    if chosen["warm_start"]:
        supervision=("using answer labels only during fine-tuning" if config["routing_auxiliary_weight"]==0
                     else f"with auxiliary matching-position loss weight {config['routing_auxiliary_weight']:g}")
        training=(f"The selected recipe initializes from a cap-64 model trained for 3000 updates, "
                  f"applies a maximum cap of 24 before any further inference, and fine-tunes for "
                  f"{config['steps']} updates at $N=8192$, {supervision}. "
                  "The parent training includes the disclosed matching-position auxiliary supervision; "
                  "its 3000 updates are an additional training resource.")
    else:
        training=(f"The selected recipe trains from scratch for {config['steps']} updates, "
                  f"with a curriculum ending at $N=8192$ and auxiliary matching-position loss "
                  f"weight {config['routing_auxiliary_weight']:g}.")
    gap=-data["cap24_minus_reference_gap_percentage_points"]
    conclusion=(f"The cap-24 result remains {gap:.3f} percentage points below the reference. "
                "This experiment therefore does not demonstrate recovery under cap 24; "
                "failure of the tested recipes is not a general impossibility result."
                if gap>0 else "The cap-24 model matches the reference on this finite test; this is specific to the stated task and training resources.")
    text=rf"""\textbf{{Additional training through $N=8192$ under cap 24.}}
We also extended the separate continuous-key, binary single-query task to
include training at $N=8192$, with $d=32$ and 8190 memory tokens per sequence.
This is a diagnostic task separate from MQAR. All six projections, including
the classifier, retain maximum cap 24; input norms, unit-norm LayerNorms,
architecture, and the native denominator $\sqrt{{32}}$ are unchanged.
We evaluated five complete three-seed cap-24 recipes, including longer
training and constrained fine-tuning, and selected a recipe by mean
$N=8192$ validation accuracy before opening the reserved final test.
{training}
Matching positions never enter inference.

On 2048 independent counterfactual pairs at $N=8192$, the selected cap-24
recipe achieves {score('selected_cap24')}, compared with
{score('reference_cap64')} for the cap-64 reference and
{score('uniform_cap24')} for the trained uniform cap-24 control
(mean $\pm$ sample standard deviation across all three seeds).
{conclusion}
All attempted recipes, training resources, cap diagnostics, and test
results are retained in the accompanying supplement.

"""
    table=[r"\begin{table}[tbp]",r"\centering\small",r"\begin{tabular}{lrrr}",r"\hline",
           r"Recipe & Total updates & Routing weight & Validation $N=8192$ (\%) \\",r"\hline"]
    names={"cap24":"Initial cap 24", "cold_long":"Longer training", "warm_route1":"Cap-64 initialization",
           "warm_route01":"Cap-64 initialization", "warm_answer":"Cap-64 initialization"}
    for r in selection["recipes"]:
        table.append(f"{names[r['recipe']]} & {r['total_updates']} & {r['routing_weight']:g} & "
                     f"${r['validation_mean_pct']:.3f}\\pm{r['validation_sd_pct']:.3f}$"+r" \\")
    table.extend([r"\hline",r"\end{tabular}",
        r"\caption{All attempted cap-24 training recipes at length 8192. Values are means and sample standard deviations across the same three seeds on validation, not final-test scores. Total updates include the 3000-update cap-64 parent for initialized recipes. The recipe with the highest three-seed mean is selected before final testing; no seed is discarded.}",
        r"\label{tab:comment4-8192-search}",r"\end{table}",""])
    supplement=(text+"\n".join(table)+r"\input{results/reviewer/long_recall_train8192_summary/results_table.tex}"+"\n\n")
    inputs={"train8192_summary_sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
    return text,supplement,inputs


def main():
    caps_root=ROOT/"results/reviewer/comment4_native_caps"
    caps=json.loads((caps_root/"summary.json").read_text())
    assert caps["audit"]["all_checks_passed"]
    lookup={(r["cap"],r["train_N"],r["variant"],r["N"]):r for r in caps["aggregates"]}
    original_recall=json.loads((ROOT/"results/reviewer/long_recall_summary/summary.json").read_text())
    old={(r["protocol"],r["train_N"],r["variant"],r["N"]):r for r in original_recall["aggregates"]}
    new1024=lookup[64,1024,"native_sqrt_d",8192]
    new2048=lookup[64,2048,"native_sqrt_d",8192]
    old1024=lookup[24,1024,"native_sqrt_d",8192]
    old2048=lookup[24,2048,"native_sqrt_d",8192]
    with (caps_root/"per_seed.csv").open(newline="") as stream:
        per_seed=list(csv.DictReader(stream))
    successful=[r for r in per_seed if r["cap"]=="64" and r["variant"]=="native_sqrt_d"]
    qk=[float(r[key]) for r in successful for key in ("q_norm","k_norm")]
    sigma1=min(float(r["minimum_sigma1"]) for r in successful)
    sigma2=min(float(r["minimum_sigma2"]) for r in successful)
    assert {int(r["active_caps"]) for r in successful}=={0}
    shuffle=[100*float(r["shuffled_query_accuracy"]) for r in successful if r["N"]=="8192"]
    extra=rf"""\textbf{{Long-context training and a matched mean-pooling control.}}
We added a classical associative-recall experiment with $d=32$, trained from
scratch using separate curricula ending at $N=1024$ or $2048$, and evaluated
through $N=8192$ ($N/d=256$). Each sequence contains $N-2$ memory tokens with
fresh continuous keys and balanced binary values, one public marker, and a query
copying a randomly selected memory key. The target is its value. Counterfactual
pairs share all keys and the query but reverse every memory value. Their row
means are exactly equal and their labels are opposite. Consequently, the
uniform-attention version of this architecture has exactly $50\%$ accuracy on
complete pairs, regardless of downstream training. This statement concerns the
specified linear value projection and downstream block, not arbitrary nonlinear
per-token pooling models.

The controlled model includes attention, both residuals and LayerNorms, a GELU
feed-forward block and a binary classifier. All six complete matrices
($W_q,W_k,W_v,M_1,M_2$ and the classifier) are trained with forward-pass
Frobenius caps. There are no learned embeddings or separate attention-output
projection. LayerNorm has fixed $\gamma_i=1/\sqrt d$, $\beta_i=0$, and zero
epsilon. Input rows have norm at most one; every training and evaluation batch
is checked in float64 for $\|S\|_F\le\sqrt N$. The evaluation includes both
native scaling $\alpha_0=\sqrt{{32}}$ and strict scaling
$\alpha_0=N C^2$, where $C$ is the stated norm cap. These settings are reported
separately.

Each cohort retains all three fixed seeds (11,22,33) and uses 3000 updates:
500 at $N=64$, 500 at $256$, and 2000 at its maximum training length. Evaluation
uses the fixed final checkpoint and 2048 independently generated counterfactual
pairs (4096 sequences) at each length. No seed or checkpoint is selected on
these test results. Initial classification-only cohorts with a cap of 24 had substantial
seed variation: for $\alpha_0=1$, accuracy at $N=8192$ was
{percentage(old['classification_only',1024,'general_alpha1',8192])} and
{percentage(old['classification_only',2048,'general_alpha1',8192])} for the two
training lengths. These initial outcomes, all other scaling variants, and every
seed are retained in Table~\ref{{tab:long-paired-recall}}.

We separately report an exploratory follow-up adding cross-entropy supervision
on the known matching-token index, with weight one. This extra signal is used
only in training; the index never enters inference. Every variant receives the
same objective, which has constant routing loss for uniform attention. Under
this protocol we compare caps of 24 and 64 while keeping the native attention
temperature, initialization, training batches, optimizer and update budget
fixed. The existing checkpoints with a cap of 24 are reevaluated without retraining on the
same fresh test stream used for the cap of 64. This follow-up was motivated by the
earlier results and is not a confirmatory protocol chosen without observing them.

At $N=8192$, the native-scaled models with a cap of 24 achieve {percentage(old1024)} and
{percentage(old2048)} for the two training lengths. The models with a cap of 64 achieve
{percentage(new1024)} and {percentage(new2048)}, respectively. Both separately
trained uniform controls and uniform-attention substitutions in the trained
models score exactly $50\%$. Shuffling query keys across independent pairs
reduces the six models with a cap of 64 to ${min(shuffle):.2f}$--${max(shuffle):.2f}\%$ accuracy.
The strict-scaled models remain approximately at $50\%$. Thus the positive
retrieval results distinguish the native-scaled models from their mean-pooling
controls; they do not demonstrate selective retrieval under $\alpha_0=N C^2$.

A cap of 64 is a larger norm allowance and must not be interpreted as FN24 recovery.
The native models with a cap of 24 have actively capped $Q/K$ matrices, whereas the
successful models with a cap of 64 have final $Q/K$ norms ranging from ${min(qk):.2f}$ to
${max(qk):.2f}$ and no active caps at their final checkpoints. Their measured
minimum LayerNorm centered norms over all evaluated lengths and seeds are
${sigma1:.4f}$ and ${sigma2:.4f}$; these are empirical minima, not uniform
lower bounds over all possible inputs. Raw/effective norms and cap activity are
retained alongside all training logs, per-example predictions, checkpoint/source
hashes and independent replay checks. Table~\ref{{tab:native-cap-recall}} reports
the complete cap comparison. These are controlled synthetic retrieval results,
separate from the descriptive pretrained SST-2 recovery experiments above.

\input{{results/reviewer/long_recall_summary/comment4_table.tex}}
\input{{results/reviewer/comment4_native_caps/native_caps_table.tex}}

"""
    previous=(ROOT/"results/reviewer/release/comment4_response.tex").read_text()
    start=previous.index(r"\textbf{Requested long-context control.}")
    end=previous.index(r"\begingroup",start)
    mqar,mqar_table,mqar_inputs=mqar_evidence()
    primary_table,seed_table,norm_table,primary_evidence=mqar_primary_evidence()
    intro=previous[:start].replace("The bounded experiments enforce", "The bounded SST-2 experiments enforce")
    intro=intro.replace("The target filter covers", "For SST-2, the target filter covers")
    intro=intro.replace("The release contains", "The accompanying SST-2 records contain")
    followup,followup_supplement,followup_inputs=training8192_evidence()
    supplement=(r"\section*{MQAR: protocol, controls, and complete results}"+"\n\n"+mqar
                +r"\clearpage\section*{SST-2: recorded protocols and recovery cohorts}"+"\n\n"
                +intro+previous[end:]
                +r"\clearpage\section*{Additional associative-recall diagnostics}"+"\n\n"
                +extra+followup_supplement)
    body=focused_response(primary_evidence)
    source=ROOT/"response.tex"
    original=source.read_text()
    comment=original.index(r"\subsection*{Comment 4}")
    marker=r"\textbf{Response}\nopagebreak[4]"
    body_start=original.index(marker,comment)+len(marker)
    next_comment=original.index(r"\subsection*{Comment 5}",body_start)
    body_end=original.rfind(r"\Needspace{6\baselineskip}",body_start,next_comment)
    assert body_end>body_start
    replacement="\n\n"+body.strip()+"\n\n\n"
    draft=original[:body_start]+replacement+original[body_end:]
    assert draft[:body_start]==original[:body_start]
    assert draft[body_start+len(replacement):]==original[body_end:]
    reviewer_pattern=r"\\begin\{reviewercomment\}.*?\\end\{reviewercomment\}"
    assert re.findall(reviewer_pattern,draft,re.S)==re.findall(reviewer_pattern,original,re.S)
    assert "quantum" not in extra.lower(),"The added Comment4 experimental text must stay classical"
    output=ROOT/"results/reviewer/comment4_update"
    output.mkdir(exist_ok=True)
    (output/"comment4_response.tex").write_text(body)
    (output/"comment4_supplement.tex").write_text(supplement)
    (output/"comment4_experiments.tex").write_text(extra)
    (output/"comment4_mqar.tex").write_text(mqar)
    (output/"comment4_mqar_table.tex").write_text(mqar_table)
    (output/"comment4_mqar_primary_table.tex").write_text(primary_table)
    (output/"comment4_mqar_seed_table.tex").write_text(seed_table)
    (output/"comment4_mqar_norm_table.tex").write_text(norm_table)
    (output/"mqar_primary_evidence.json").write_text(json.dumps(primary_evidence,indent=2)+"\n")
    (output/"response_comment4_only.tex").write_text(draft)
    preamble=original[:original.index(r"\begin{document}")]
    preamble=preamble.replace(r"\title{Response to the second-round reports}",
                              r"\title{Response to Reviewer 1: Comment 4}")
    standalone=(preamble+r"\begin{document}"+"\n"+r"\maketitle"+"\n\n"
                +original[comment:body_start]+replacement+r"\end{document}"+"\n")
    def inline_tables(match):
        path=ROOT/match.group(1)
        return path.read_text()
    standalone=re.sub(r"\\input\{([^}]+)\}",inline_tables,standalone)
    assert r"\input{" not in standalone
    (output/"comment4_only.tex").write_text(standalone)
    supplement_preamble=preamble.replace(r"\title{Response to Reviewer 1: Comment 4}",
                                         r"\title{Experimental supplement for Comment 4}")
    supplement_standalone=(supplement_preamble+r"\begin{document}"+"\n"+r"\maketitle"+"\n\n"
                           +supplement+r"\end{document}"+"\n")
    supplement_standalone=re.sub(r"\\input\{([^}]+)\}",inline_tables,supplement_standalone)
    assert r"\input{" not in supplement_standalone
    (output/"comment4_supplement_standalone.tex").write_text(supplement_standalone)
    report={"original_sha256":hashlib.sha256(original.encode()).hexdigest(),
        "draft_sha256":hashlib.sha256(draft.encode()).hexdigest(),"changed_response_bodies":[4],
        "assembler_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "all_other_document_text_unchanged":True,"reviewer_comments_unchanged":True,
        "original_file_unchanged":source.read_text()==original,"quantum_additions":False,
        "latex_compilation":"Not performed; no TeX compiler available",
        "inputs":{"native_cap_summary_sha256":hashlib.sha256((caps_root/"summary.json").read_bytes()).hexdigest(),
                  "original_recall_summary_sha256":hashlib.sha256((ROOT/"results/reviewer/long_recall_summary/summary.json").read_bytes()).hexdigest(),
                  "mqar_primary_evidence_sha256":hashlib.sha256((output/"mqar_primary_evidence.json").read_bytes()).hexdigest(),
                  **mqar_inputs,**followup_inputs}}
    (output/"provenance.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
