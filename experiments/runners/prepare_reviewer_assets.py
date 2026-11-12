"""Download public, revision-pinned assets for the reviewer experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--dataset-revision", default=None)
    args = parser.parse_args()
    api = HfApi(token=False)
    model_id = "EleutherAI/gpt-neo-125m"
    dataset_id = "Salesforce/wikitext"
    model_info = api.model_info(model_id, revision=args.model_revision)
    dataset_info = api.dataset_info(dataset_id, revision=args.dataset_revision)
    files = {item.rfilename for item in model_info.siblings}
    weights = "model.safetensors" if "model.safetensors" in files else "pytorch_model.bin"
    model_files = [weights, "config.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt", "tokenizer.json"]
    data_files = [item.rfilename for item in dataset_info.siblings
                  if item.rfilename.startswith("wikitext-2-raw-v1/validation-") and item.rfilename.endswith(".parquet")]
    if not data_files:
        raise RuntimeError("No WikiText-2 raw validation parquet found")
    args.output.mkdir(parents=True, exist_ok=True)
    snapshot_download(model_id, revision=model_info.sha, token=False,
                      allow_patterns=model_files, local_dir=args.output / "model", max_workers=2)
    snapshot_download(dataset_id, repo_type="dataset", revision=dataset_info.sha, token=False,
                      allow_patterns=data_files, local_dir=args.output / "data", max_workers=2)
    manifest = {"model_id": model_id, "model_revision": model_info.sha,
                "dataset_id": dataset_id, "dataset_revision": dataset_info.sha,
                "dataset_subset": "wikitext-2-raw-v1", "split": "validation",
                "weights_file": weights, "data_files": data_files}
    (args.output / "assets.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
