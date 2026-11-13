"""Pin, tokenize and audit data before any experiment is trained."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def duplicate_groups(raw_keys, token_keys):
    """Connect examples sharing either full input or effective tokenized input."""
    parents = list(range(len(raw_keys)))

    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    for keys in (raw_keys, token_keys):
        seen = {}
        for i, key in enumerate(keys):
            if key in seen:
                a, b = root(i), root(seen[key])
                parents[max(a, b)] = min(a, b)
            else:
                seen[key] = i
    return [root(i) for i in range(len(raw_keys))]


def clean_splits(original, groups, labels, make_validation=False, split_seed=20261112):
    """Keep test intact; remove matching groups from lower priority splits."""
    test_groups = {groups[i] for i, s in enumerate(original) if s == "test"}
    dev_groups = {groups[i] for i, s in enumerate(original) if s == "validation"} - test_groups
    selected = {"train": [], "validation": [], "test": []}
    removed = []
    for i, split in enumerate(original):
        if split == "test":
            selected[split].append(i)
        elif groups[i] in test_groups or (split == "train" and groups[i] in dev_groups):
            removed.append(i)
        else:
            selected[split].append(i)
    if make_validation:
        by_group = defaultdict(list)
        for i in selected["train"]:
            by_group[groups[i]].append(i)
        group_ids = sorted(by_group)
        # Mixed-label groups stay together; majority label is used only for stratification.
        strata = [Counter(int(labels[i]) for i in by_group[g]).most_common(1)[0][0] for g in group_ids]
        _, validation = train_test_split(group_ids, test_size=0.1, random_state=split_seed, stratify=strata)
        validation = set(validation)
        selected["validation"] = [i for i in selected["train"] if groups[i] in validation]
        selected["train"] = [i for i in selected["train"] if groups[i] not in validation]
    group_sets = {s: {groups[i] for i in rows} for s, rows in selected.items()}
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if group_sets[a] & group_sets[b]:
            raise AssertionError(f"Duplicate groups cross {a}/{b}")
    return selected, removed


def prepare_one(name, sources, destination, llama_path):
    maximum = {"genomic": 72, "gue": 96, "conll": 64}[name]
    is_tokens = name == "conll"
    model_path = llama_path if is_tokens else sources["zhihan1996/DNABERT-2-117M"]["snapshot_path"]
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    tokenizer.padding_side = "right"
    if is_tokens:
        tokenizer.pad_token = tokenizer.eos_token
    records, source_files, label_names = [], {}, None
    repository = {"genomic": "katarinagresova/Genomic_Benchmarks_human_nontata_promoters",
                  "gue": "leannmlindsey/GUE", "conll": "eriktks/conll2003"}[name]
    root = Path(sources[repository]["snapshot_path"])
    for split in ("train", "validation", "test"):
        if name == "genomic":
            if split == "validation":
                continue
            path = next(root.glob(f"data/{split}-*.parquet"))
        elif name == "gue":
            path = root / "GUE/prom_300_notata" / f"{'dev' if split == 'validation' else split}.csv"
        else:
            path = root / "conll2003" / split / "0000.parquet"
            meta = json.loads(pq.read_schema(path).metadata[b"huggingface"])
            label_names = meta["info"]["features"]["pos_tags"]["feature"]["names"]
        source_files[str(path)] = file_sha(path)
        df = pd.read_csv(path) if path.suffix == ".csv" else pd.read_parquet(path)
        for i, row in enumerate(df.to_dict("records")):
            value = list(row["tokens"]) if is_tokens else row["seq" if name == "genomic" else "sequence"].upper()
            labels = list(map(int, row["pos_tags"])) if is_tokens else int(row["label"])
            records.append({"id": f"{name}:{split}:{i}", "split": split, "input": value, "label": labels})
    input_ids, masks, labels, raw_keys, token_keys, truncated = [], [], [], [], [], []
    for start in range(0, len(records), 512):
        rows = records[start:start + 512]
        text = [r["input"] for r in rows]
        full = tokenizer(text, is_split_into_words=is_tokens, truncation=False, padding=False)
        encoded = tokenizer(text, is_split_into_words=is_tokens, truncation=True, padding="max_length", max_length=maximum)
        for j, row in enumerate(rows):
            ids, mask = encoded["input_ids"][j], encoded["attention_mask"][j]
            input_ids.append(ids)
            masks.append(mask)
            truncated.append(len(full["input_ids"][j]) > maximum)
            if is_tokens:
                word_ids, previous, aligned = encoded.word_ids(batch_index=j), None, []
                for word in word_ids:
                    aligned.append(row["label"][word] if word is not None and word != previous else -100)
                    previous = word
                labels.append(aligned)
                raw_key = row["input"]
            else:
                labels.append(row["label"])
                sequence = row["input"]
                complement = sequence.translate(str.maketrans("ACGTN", "TGCAN"))[::-1]
                raw_key = min(sequence, complement)
            raw_keys.append(digest(raw_key))
            token_keys.append(digest([v for v, active in zip(ids, mask) if active]))
    groups = duplicate_groups(raw_keys, token_keys)
    selected, removed = clean_splits([r["split"] for r in records], groups, labels, make_validation=name == "genomic")
    destination.mkdir(parents=True, exist_ok=True)
    report = {"dataset": name, "task": "POS tagging" if is_tokens else "binary classification",
              "source": sources[repository], "source_file_sha256": source_files,
              "tokenizer_path": model_path, "max_length": maximum, "num_labels": len(label_names) if label_names else 2,
              "label_names": label_names, "split_seed": 20261112, "pad_token_id": tokenizer.pad_token_id,
              "duplicate_rule": "connected components of full input (DNA reverse complement canonicalized) and effective token IDs",
              "priority": "test > official validation > train; genomic validation is 10% stratified training groups",
              "removed_overlap_ids": [records[i]["id"] for i in removed], "splits": {},
              "limitations": ["Exact duplication checks do not establish biological homology independence.",
                              "All official test rows are retained, including repeated inputs."]}
    for split, indices in selected.items():
        data = {"input_ids": torch.tensor([input_ids[i] for i in indices], dtype=torch.long),
                "attention_mask": torch.tensor([masks[i] for i in indices], dtype=torch.long),
                "labels": torch.tensor([labels[i] for i in indices], dtype=torch.long)}
        path = destination / f"{split}.pt"
        torch.save(data, path)
        metadata = [{"id": records[i]["id"], "group": groups[i], "raw_sha256": raw_keys[i],
                     "tokens_sha256": token_keys[i], "truncated": truncated[i]} for i in indices]
        (destination / f"{split}_rows.json").write_text(json.dumps(metadata))
        counts = Counter(data["labels"][data["labels"] != -100].tolist())
        report["splits"][split] = {"rows": len(indices), "groups": len({groups[i] for i in indices}),
                                   "label_counts": dict(counts), "truncated": sum(truncated[i] for i in indices),
                                   "tensor_sha256": file_sha(path), "row_manifest_sha256": file_sha(destination / f"{split}_rows.json")}
    report["manifest_sha256"] = digest(report)
    (destination / "manifest.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"dataset": name, "splits": report["splits"], "overlaps_removed": len(removed)}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--llama-path", required=True)
    args = parser.parse_args()
    sources = json.loads(args.sources.read_text())
    for name in ("genomic", "gue", "conll"):
        if (args.output / name / "manifest.json").exists():
            raise FileExistsError(f"Prepared data already exists: {name}")
        prepare_one(name, sources, args.output / name, args.llama_path)


if __name__ == "__main__":
    main()
