"""Describe MQAR runs using their saved configuration."""

import json
import math


def cohort_config(root):
    paths = sorted(root.glob("native_seed*/config.json"))
    if not paths:
        raise FileNotFoundError(f"No completed native-attention configurations in {root}")
    config = json.loads(paths[0].read_text())
    for path in paths[1:]:
        if json.loads(path.read_text()) != config:
            raise ValueError(f"Inconsistent cohort configuration: {path}")
    return config


def protocol_summary(config):
    updates = math.ceil(config["train_examples"] / config["batch_size"]) * config["epochs"]
    description = (
        f"Vocabulary {config['vocab_size']:,}; training N={config['train_length']}, "
        f"{config['train_pairs']} memory pairs, {config['train_examples']:,} sequences; "
        f"{config['validation_examples']:,} validation and {config['test_examples']:,} test sequences per condition. "
        f"{config['layers']} layers, d={config['dimension']}, FFN width {config['ffn_dimension']}; "
        f"{config['epochs']} epochs, {updates:,} optimizer updates; seeds {config['seeds']}."
    )
    if config.get("workflow_validation"):
        description = "WORKFLOW VALIDATION ONLY; these are not benchmark reproduction scores. " + description
    return description
