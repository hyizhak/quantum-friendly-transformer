#!/usr/bin/env python
"""Evaluate one-at-a-time FN24 lowering for remaining BERT intermediate matrices."""

from __future__ import annotations

import argparse
import os
import re
import time

from quantum_friendly_transformer.norm_transformer.frobenius_cap import (
    apply_learnable_frobenius_caps,
    collect_learnable_cap_stats,
)

from quantum_friendly_transformer.experiments.lib.common import (
    ensure_parent,
    now_metadata,
    package_versions,
    seed_everything,
    write_json,
)
from quantum_friendly_transformer.experiments.runners.recovery.glue_distill import (
    load_student_state_with_caps,
    set_learnable_cap_max_for_bert_layer_module_group,
)
from quantum_friendly_transformer.experiments.runners.recovery.glue_sequence_classifier import (
    GLUE_TEXT_COLUMNS,
    cap_summary,
    evaluate,
    load_model,
    make_loaders,
    pick_device,
    prepare_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--task", required=True, choices=sorted(GLUE_TEXT_COLUMNS))
    parser.add_argument("--student-state-in", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--pad-to-max-length", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--cap-norm-dtype", choices=["native", "float32"], default="float32")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--force-decoder-wrapper", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--bnb-4bit-quant-type", default="nf4", choices=["nf4", "fp4"])
    parser.add_argument("--bnb-4bit-compute-dtype", choices=["bfloat16", "float16", "float32"], default=None)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--offline", action="store_true")
    return parser.parse_args()


def remaining_intermediate_layers(cap_max_gamma_by_name: dict[str, float]) -> list[int]:
    layers: set[int] = set()
    for name, cap in cap_max_gamma_by_name.items():
        if not name.endswith("intermediate.dense.weight"):
            continue
        if float(cap) <= 24.0001:
            continue
        match = re.search(r"encoder\.layer\.(\d+)\.", name)
        if match:
            layers.add(int(match.group(1)))
    return sorted(layers)


def main() -> None:
    args = parse_args()
    import torch
    from datasets import load_dataset
    from transformers import AutoTokenizer

    seed_everything(args.seed)
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"

    device = pick_device(torch, args.device)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=True,
        cache_dir=args.cache_dir,
        local_files_only=args.offline,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    tokenizer.padding_side = "right"

    train_ds, eval_ds = prepare_dataset(args, tokenizer, load_dataset)
    _, eval_loader = make_loaders(torch, args, tokenizer, train_ds, eval_ds)

    model = load_model(torch, args, tokenizer, device)
    apply_learnable_frobenius_caps(
        model,
        64.0,
        init_fraction=0.999,
        norm_dtype=args.cap_norm_dtype,
    )

    state = torch.load(args.student_state_in, map_location="cpu")
    if not isinstance(state, dict) or "state_dict" not in state:
        raise ValueError("--student-state-in must be a checkpoint with state_dict and cap metadata")
    cap_map = state.get("cap_max_gamma_by_name") or {}
    if not cap_map:
        raise ValueError("--student-state-in is missing cap_max_gamma_by_name")

    layers = remaining_intermediate_layers(cap_map)
    rows = []
    start = time.time()

    def evaluate_current(label: str, lowered_layer: int | None) -> None:
        metrics = evaluate(torch, model, eval_loader, device, args.task)
        caps = collect_learnable_cap_stats(model)
        summary = cap_summary(caps)
        row = {
            "label": label,
            "lowered_layer": lowered_layer,
            "metric": "accuracy",
            "metric_value": metrics.get("accuracy"),
            "metrics": metrics,
            "cap_summary": summary,
            "runtime_sec": time.time() - start,
        }
        rows.append(row)
        print(
            f"{label} layer={lowered_layer} accuracy={row['metric_value']:.6g} "
            f"cap_min={summary.get('gamma_min'):.6g} cap_max={summary.get('gamma_max'):.6g}",
            flush=True,
        )

    load_student_state_with_caps(model, state)
    model.to(device)
    evaluate_current("base_partcap", None)

    for layer in layers:
        load_student_state_with_caps(model, state)
        model.to(device)
        updated = set_learnable_cap_max_for_bert_layer_module_group(
            model,
            24.0,
            [layer],
            ["intermediate"],
            init_fraction=None,
        )
        if updated != 1:
            raise RuntimeError(f"expected one updated parametrization for layer {layer}, got {updated}")
        evaluate_current("single_intermediate_to_24", layer)

    payload = {
        **now_metadata(),
        "model": args.model,
        "task": args.task,
        "dataset": "glue",
        "student_state_in": args.student_state_in,
        "screen": "single_remaining_intermediate_to_fn24_eval_only",
        "remaining_intermediate_layers": layers,
        "rows": rows,
        "package_versions": package_versions(),
    }
    output = ensure_parent(args.output)
    write_json(output, payload)
    print(f"wrote {output}", flush=True)


if __name__ == "__main__":
    main()
