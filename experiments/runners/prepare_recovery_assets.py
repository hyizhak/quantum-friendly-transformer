"""Download pinned public assets for the bounded recovery extension."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=(Path(__file__).resolve().parents[1] / 'configs/recovery/recovery_extension.json'))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    for name, model in config["models"].items():
        snapshot_download(model["model_id"], revision=model["revision"], token=False,
                          allow_patterns=["config.json", "pytorch_model.bin", "model.safetensors",
                                          "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                                          "vocab.json", "merges.txt"],
                          local_dir=args.output / name, max_workers=2)
    snapshot_download(config["dataset_id"], repo_type="dataset", revision=config["dataset_revision"],
                      token=False, allow_patterns=["sst2/train-*.parquet", "sst2/validation-*.parquet"],
                      local_dir=args.output / "data", max_workers=2)
    files = {}
    for path in sorted(args.output.rglob("*")):
        if path.is_file() and ".cache" not in path.parts and path.name != "assets.json":
            files[str(path.relative_to(args.output))] = hashlib.sha256(path.read_bytes()).hexdigest()
    record = {"dataset_id": config["dataset_id"], "dataset_revision": config["dataset_revision"],
              "models": {name: {k: m[k] for k in ["model_id", "revision"]} for name, m in config["models"].items()},
              "sha256": files}
    (args.output / "assets.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"Prepared {len(files)} pinned assets in {args.output}")


if __name__ == "__main__":
    main()
