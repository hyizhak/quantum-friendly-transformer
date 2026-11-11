import argparse
import math
import os
import time

from quantum_friendly_transformer.experiments.lib.common import (
    collect_weight_stats,
    ensure_parent,
    metadata_path,
    now_metadata,
    package_versions,
    parse_variant,
    project_model_weights,
    seed_everything,
    write_json,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run WikiText causal-LM normalisation experiments.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", default="wikitext")
    parser.add_argument("--config", default="wikitext-103-raw-v1")
    parser.add_argument("--variants", nargs="+", required=True)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--train-steps", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-train-tokens", type=int, default=None)
    parser.add_argument("--max-eval-tokens", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def pick_device(torch, requested):
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def wikitext_prefix(dataset, max_tokens):
    if max_tokens is None:
        return "\n\n".join(row["text"] for row in dataset if row["text"].strip())
    char_budget = max(4096, int(max_tokens) * 8)
    parts = []
    total_chars = 0
    for row in dataset:
        piece = row["text"].strip()
        if not piece:
            continue
        parts.append(piece)
        total_chars += len(piece) + 2
        if total_chars >= char_budget:
            break
    return "\n\n".join(parts)


def chunk_records(tokenizer, load_dataset, config, split, seq_len, max_tokens, cache_dir):
    dataset = load_dataset("wikitext", config, split=split, cache_dir=cache_dir)
    text = wikitext_prefix(dataset, max_tokens)
    tokenize_kwargs = {"add_special_tokens": False}
    if max_tokens is not None:
        tokenize_kwargs.update({"truncation": True, "max_length": max_tokens})
    ids = tokenizer(text, **tokenize_kwargs)["input_ids"]
    if max_tokens is not None:
        ids = ids[:max_tokens]
    records = []
    for start in range(0, len(ids) - seq_len + 1, seq_len):
        chunk = ids[start : start + seq_len]
        records.append({"input_ids": chunk, "attention_mask": [1] * len(chunk), "labels": chunk.copy()})
    if not records:
        raise ValueError(f"no full chunks for split={split} seq_len={seq_len}")
    return records


def make_loader(torch, records, batch_size, shuffle=False):
    from torch.utils.data import DataLoader

    def collate(batch):
        return {
            "input_ids": torch.tensor([x["input_ids"] for x in batch], dtype=torch.long),
            "attention_mask": torch.tensor([x["attention_mask"] for x in batch], dtype=torch.long),
            "labels": torch.tensor([x["labels"] for x in batch], dtype=torch.long),
        }

    return DataLoader(records, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)


def move_batch(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def evaluate(torch, model, loader, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            outputs = model(**batch)
            labels = batch["labels"]
            # Hugging Face causal LM loss is averaged over shifted non-ignored labels.
            tokens = int((labels[:, 1:] != -100).sum().item())
            total_loss += float(outputs.loss.detach().cpu().item()) * max(1, tokens)
            total_tokens += max(1, tokens)
    loss = total_loss / max(1, total_tokens)
    return {"loss": loss, "perplexity": float(math.exp(min(50.0, loss))), "tokens": total_tokens}


def train_steps(torch, model, loader, device, variant, steps, lr, weight_decay):
    if steps <= 0:
        return {"train_steps": 0, "mean_train_loss": None}
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    losses = []
    model.train()
    step_count = 0
    while step_count < steps:
        for batch in loader:
            batch = move_batch(batch, device)
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if variant.kind != "baseline":
                project_model_weights(model, variant.kind, variant.tau)
            losses.append(float(loss.detach().cpu().item()))
            step_count += 1
            if step_count >= steps:
                break
    return {"train_steps": step_count, "mean_train_loss": sum(losses) / max(1, len(losses))}


def run_variant(torch, args, variant, train_loader, eval_loader, device):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        cache_dir=args.cache_dir,
        local_files_only=args.offline,
    )
    model.to(device)
    before_stats = collect_weight_stats(model)
    projected_after_load = 0
    if variant.kind != "baseline":
        projected_after_load = project_model_weights(model, variant.kind, variant.tau)
    after_load_stats = collect_weight_stats(model)

    start = time.time()
    train_info = {"train_steps": 0, "mean_train_loss": None}
    if args.train_steps > 0 or variant.recovery:
        if train_loader is None:
            raise ValueError("training or recovery requested, but no training loader was built")
        steps = args.train_steps if args.train_steps > 0 else 2000
        train_info = train_steps(
            torch,
            model,
            train_loader,
            device,
            variant,
            steps,
            args.learning_rate,
            args.weight_decay,
        )
    metrics = evaluate(torch, model, eval_loader, device)
    return {
        "variant": variant.raw,
        "kind": variant.kind,
        "tau": variant.tau,
        "recovery": variant.recovery,
        "projected_after_load": projected_after_load,
        "runtime_sec": time.time() - start,
        "metrics": metrics,
        **train_info,
        "weight_stats_before": before_stats,
        "weight_stats_after_load": after_load_stats,
        "weight_stats_final": collect_weight_stats(model),
    }


def main():
    args = parse_args()
    import torch
    from datasets import load_dataset
    from transformers import AutoTokenizer

    seed_everything(args.seed)
    device = pick_device(torch, args.device)
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=True,
        cache_dir=args.cache_dir,
        local_files_only=args.offline,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    variants = [parse_variant(spec) for spec in args.variants]
    need_train = args.train_steps > 0 or any(variant.recovery for variant in variants)
    train_loader = None
    if need_train:
        train_records = chunk_records(
            tokenizer,
            load_dataset,
            args.config,
            "train",
            args.seq_len,
            args.max_train_tokens or args.max_eval_tokens or args.seq_len * args.batch_size * 8,
            args.cache_dir,
        )
        train_loader = make_loader(torch, train_records, args.batch_size, shuffle=True)
    eval_records = chunk_records(
        tokenizer, load_dataset, args.config, "validation", args.seq_len, args.max_eval_tokens, args.cache_dir
    )
    eval_loader = make_loader(torch, eval_records, args.batch_size, shuffle=False)

    results = []
    for variant in variants:
        seed_everything(args.seed)
        print(f"running {variant.raw} on WikiText", flush=True)
        results.append(run_variant(torch, args, variant, train_loader, eval_loader, device))

    output = ensure_parent(args.output)
    payload = {
        **now_metadata(),
        "model": args.model,
        "dataset": args.dataset,
        "config": args.config,
        "split": "validation",
        "seed": args.seed,
        "device": str(device),
        "cache_dir": args.cache_dir,
        "offline": args.offline,
        "seq_len": args.seq_len,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "max_train_tokens": args.max_train_tokens,
        "max_eval_tokens": args.max_eval_tokens,
        "package_versions": package_versions(),
        "results": results,
    }
    write_json(output, payload)
    write_json(metadata_path(output), {k: v for k, v in payload.items() if k != "results"})
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
