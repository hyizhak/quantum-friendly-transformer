import argparse
import gc
import os
from collections import defaultdict
from pathlib import Path

from quantum_friendly_transformer.experiments.runners.diagnostics.forward_stats import build_wikitext_records, make_batches


def parse_args():
    parser = argparse.ArgumentParser(description="Compute exact global LayerNorm quantiles by norm slot.")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=200000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--offline", action="store_true")
    return parser.parse_args()


def norm_modules(model):
    model_type = getattr(model.config, "model_type", None)
    if model_type == "gpt2":
        base = getattr(model, "transformer", model)
        return [(idx, block.ln_1, block.ln_2, False) for idx, block in enumerate(base.h)]
    if model_type == "opt":
        base = getattr(model, "decoder", getattr(model, "model", model))
        return [
            (idx, layer.self_attn_layer_norm, layer.final_layer_norm, False)
            for idx, layer in enumerate(base.layers)
        ]
    if model_type in {"qwen2", "qwen3"}:
        base = getattr(model, "model", model)
        return [
            (idx, layer.input_layernorm, layer.post_attention_layernorm, True)
            for idx, layer in enumerate(base.layers)
        ]
    raise ValueError(f"unsupported model type for norm collection: {model_type}")


def collect_model(torch, np, load_dataset, AutoModel, AutoTokenizer, args, model_name):
    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        cache_dir=args.cache_dir,
        local_files_only=args.offline,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    records = build_wikitext_records(
        load_dataset,
        tokenizer,
        "wikitext-103-raw-v1",
        "validation",
        args.seq_len,
        None,
        args.max_tokens,
        args.cache_dir,
    )
    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=dtype,
        cache_dir=args.cache_dir,
        local_files_only=args.offline,
        low_cpu_mem_usage=True,
        device_map={"": "cuda:0"},
    )
    model.eval()

    batch_values = defaultdict(list)
    pooled = defaultdict(list)
    handles = []

    def register(module, slot, rms):
        def hook(mod, inputs):
            x = inputs[0].detach().float()
            eps = getattr(mod, "variance_epsilon", None) or getattr(mod, "eps", 1e-5)
            if rms:
                denom = torch.sqrt(torch.mean(x * x, dim=-1) + eps)
            else:
                centered = x - x.mean(dim=-1, keepdim=True)
                denom = torch.sqrt(torch.mean(centered * centered, dim=-1) + eps)
            batch_values[slot].append(denom)

        handles.append(module.register_forward_pre_hook(hook))

    modules = norm_modules(model)
    for _, norm1, norm2, rms in modules:
        register(norm1, "norm1", rms)
        register(norm2, "norm2", rms)

    try:
        with torch.inference_mode():
            total_batches = (len(records) + args.batch_size - 1) // args.batch_size
            for batch_index, batch in enumerate(make_batches(torch, records, args.batch_size), start=1):
                batch_values.clear()
                inputs = {
                    "input_ids": batch["input_ids"].to("cuda:0"),
                    "attention_mask": batch["attention_mask"].to("cuda:0"),
                }
                outputs = model(**inputs, use_cache=False, return_dict=True)
                del outputs
                for slot, tensors in batch_values.items():
                    pooled[slot].append(torch.cat([tensor.reshape(-1) for tensor in tensors]).cpu().numpy())
                if batch_index % 10 == 0 or batch_index == total_batches:
                    print(f"{model_name}: {batch_index}/{total_batches} batches", flush=True)
    finally:
        for handle in handles:
            handle.remove()

    rows = []
    for slot in ["norm1", "norm2"]:
        values = np.concatenate(pooled[slot])
        p1, p5, median, p95 = np.quantile(values, [0.01, 0.05, 0.50, 0.95])
        rows.append(
            {
                "model": model_name,
                "family": args.family,
                "norm_type": "RMSNorm" if modules[0][3] else "LayerNorm",
                "dataset": "wikitext-103-raw-v1",
                "seq_len": args.seq_len,
                "d": int(getattr(model.config, "hidden_size")),
                "num_layers": len(modules),
                "norm_slot": slot,
                "p1": float(p1),
                "p5": float(p5),
                "median": float(median),
                "p95": float(p95),
                "sample_count": int(values.size),
                "quantile_scope": "global_within_norm_slot",
            }
        )

    del model, tokenizer, records, pooled, batch_values
    gc.collect()
    torch.cuda.empty_cache()
    return rows


def main():
    args = parse_args()
    import numpy as np
    import pandas as pd
    import torch
    from datasets import load_dataset
    from transformers import AutoModel, AutoTokenizer

    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the retained d-scaling model set")

    rows = []
    for model_name in args.models:
        rows.extend(collect_model(torch, np, load_dataset, AutoModel, AutoTokenizer, args, model_name))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"wrote {output}", flush=True)


if __name__ == "__main__":
    main()
