import argparse
import os
import time

from quantum_friendly_transformer.experiments.lib.common import (
    collect_weight_stats,
    compute_classification_metrics,
    ensure_parent,
    glue_num_labels,
    glue_split_name,
    metadata_path,
    now_metadata,
    package_versions,
    parse_variant,
    project_model_weights,
    seed_everything,
    write_json,
)


GLUE_TEXT_COLUMNS = {
    "cola": ("sentence", None),
    "sst2": ("sentence", None),
    "mrpc": ("sentence1", "sentence2"),
    "qqp": ("question1", "question2"),
    "stsb": ("sentence1", "sentence2"),
    "mnli": ("premise", "hypothesis"),
    "qnli": ("question", "sentence"),
    "rte": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run GLUE normalisation robustness experiments.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--task", required=True, choices=sorted(GLUE_TEXT_COLUMNS))
    parser.add_argument("--variants", nargs="+", required=True)
    parser.add_argument("--projection", choices=["after_load", "after_step"], default="after_step")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
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


def prepare_dataset(args, tokenizer, load_dataset):
    train = load_dataset("glue", args.task, split="train", cache_dir=args.cache_dir)
    eval_split = glue_split_name(args.task, "validation")
    eval_ds = load_dataset("glue", args.task, split=eval_split, cache_dir=args.cache_dir)
    if args.max_train_samples is not None:
        train = train.select(range(min(args.max_train_samples, len(train))))
    if args.max_eval_samples is not None:
        eval_ds = eval_ds.select(range(min(args.max_eval_samples, len(eval_ds))))

    col1, col2 = GLUE_TEXT_COLUMNS[args.task]

    def tokenize(batch):
        if col2 is None:
            encoded = tokenizer(batch[col1], truncation=True)
        else:
            encoded = tokenizer(batch[col1], batch[col2], truncation=True)
        encoded["labels"] = batch["label"]
        return encoded

    keep = {"input_ids", "attention_mask", "token_type_ids", "labels"}
    train = train.map(tokenize, batched=True)
    eval_ds = eval_ds.map(tokenize, batched=True)
    train = train.remove_columns([c for c in train.column_names if c not in keep])
    eval_ds = eval_ds.remove_columns([c for c in eval_ds.column_names if c not in keep])
    return train, eval_ds


def make_loaders(torch, args, tokenizer, train_ds, eval_ds):
    from torch.utils.data import DataLoader
    from transformers import DataCollatorWithPadding

    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_loader = DataLoader(train_ds, shuffle=True, batch_size=args.batch_size, collate_fn=collator)
    eval_loader = DataLoader(eval_ds, shuffle=False, batch_size=args.batch_size, collate_fn=collator)
    return train_loader, eval_loader


def move_batch(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def evaluate(torch, model, loader, device, task):
    model.eval()
    losses = []
    preds = []
    labels = []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            outputs = model(**batch)
            losses.append(float(outputs.loss.detach().cpu().item()) * batch["labels"].shape[0])
            logits = outputs.logits.detach().cpu()
            label = batch["labels"].detach().cpu()
            if task == "stsb":
                pred = logits.squeeze(-1).numpy().tolist()
            else:
                pred = logits.argmax(dim=-1).numpy().tolist()
            preds.extend(pred)
            labels.extend(label.numpy().tolist())
    metrics = compute_classification_metrics(task, preds, labels)
    metrics["eval_loss"] = sum(losses) / max(1, len(labels))
    return metrics


def train_variant(torch, args, variant, tokenizer, train_loader, eval_loader, device):
    from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup

    num_labels = glue_num_labels(args.task)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=num_labels,
        cache_dir=args.cache_dir,
        local_files_only=args.offline,
    )
    if args.task == "stsb":
        model.config.problem_type = "regression"
    model.to(device)

    before_stats = collect_weight_stats(model)
    projected_after_load = 0
    if variant.kind != "baseline":
        projected_after_load = project_model_weights(model, variant.kind, variant.tau)
    after_load_stats = collect_weight_stats(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = max(1, int(args.epochs * len(train_loader)))
    warmup_steps = int(args.warmup_ratio * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    start = time.time()
    global_step = 0
    train_losses = []
    full_epochs = int(args.epochs)
    fractional = args.epochs - full_epochs
    epoch_budgets = [len(train_loader)] * full_epochs
    if fractional > 0:
        epoch_budgets.append(max(1, int(fractional * len(train_loader))))

    for budget in epoch_budgets:
        model.train()
        for step, batch in enumerate(train_loader):
            if step >= budget:
                break
            batch = move_batch(batch, device)
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            if variant.kind != "baseline" and args.projection == "after_step":
                project_model_weights(model, variant.kind, variant.tau)
            train_losses.append(float(loss.detach().cpu().item()))
            global_step += 1
            if global_step >= total_steps:
                break
        if global_step >= total_steps:
            break

    metrics = evaluate(torch, model, eval_loader, device, args.task)
    final_stats = collect_weight_stats(model)
    return {
        "variant": variant.raw,
        "kind": variant.kind,
        "tau": variant.tau,
        "recovery": variant.recovery,
        "projection": args.projection,
        "projected_after_load": projected_after_load,
        "train_steps": global_step,
        "mean_train_loss": sum(train_losses) / max(1, len(train_losses)),
        "runtime_sec": time.time() - start,
        "metrics": metrics,
        "weight_stats_before": before_stats,
        "weight_stats_after_load": after_load_stats,
        "weight_stats_final": final_stats,
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
    train_ds, eval_ds = prepare_dataset(args, tokenizer, load_dataset)
    train_loader, eval_loader = make_loaders(torch, args, tokenizer, train_ds, eval_ds)

    variants = [parse_variant(spec) for spec in args.variants]
    results = []
    for variant in variants:
        seed_everything(args.seed)
        print(f"running {variant.raw} on {args.task}", flush=True)
        results.append(train_variant(torch, args, variant, tokenizer, train_loader, eval_loader, device))

    output = ensure_parent(args.output)
    payload = {
        **now_metadata(),
        "model": args.model,
        "task": args.task,
        "dataset": "glue",
        "split": glue_split_name(args.task, "validation"),
        "seed": args.seed,
        "device": str(device),
        "cache_dir": args.cache_dir,
        "offline": args.offline,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "max_train_samples": args.max_train_samples,
        "max_eval_samples": args.max_eval_samples,
        "package_versions": package_versions(),
        "results": results,
    }
    write_json(output, payload)
    write_json(metadata_path(output), {k: v for k, v in payload.items() if k != "results"})
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
