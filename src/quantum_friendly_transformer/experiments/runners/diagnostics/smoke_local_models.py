import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
from transformers import BertConfig, BertModel, GPT2Config, GPT2Model, LlamaConfig, LlamaModel

from quantum_friendly_transformer.experiments.runners.diagnostics.forward_stats import ForwardStatHooks, append_attention_rows


def collect_for_model(model, input_ids, attention_mask, context):
    model.eval()
    hooks = ForwardStatHooks(model, torch)
    rows = []
    sample_counts = defaultdict(int)
    try:
        with torch.no_grad():
            hooks.clear()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            append_attention_rows(
                torch,
                rows,
                sample_counts,
                hooks,
                {"attention_mask": attention_mask, "sample_index": list(range(input_ids.shape[0]))},
                outputs,
                context,
                sample_rows_per_layer_head=8,
                spectral_iters=2,
            )
    finally:
        hooks.close()
    return rows


def main():
    parser = argparse.ArgumentParser(description="Collect diagnostics from tiny random CPU models without downloads.")
    parser.add_argument("--output", type=Path, default=Path("experiments/outputs/raw/smoke_local_diagnostics.parquet"))
    args = parser.parse_args()
    torch.manual_seed(42)
    bert = BertModel(
        BertConfig(
            vocab_size=32,
            hidden_size=16,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=32,
            max_position_embeddings=32,
        )
    )
    bert_rows = collect_for_model(
        bert,
        torch.randint(1, 31, (2, 12)),
        torch.tensor([[1] * 10 + [0, 0], [1] * 12]),
        {
            "model": "local-random-bert",
            "dataset": "synthetic",
            "task": "smoke",
            "split": "none",
            "seq_len": 12,
            "model_type": "bert",
            "num_layers": 2,
            "hidden_dim": 16,
            "num_heads": 4,
            "norm_type": "LayerNorm",
            "attention_type": "bidirectional",
        },
    )

    gpt2 = GPT2Model(
        GPT2Config(
            vocab_size=32,
            n_embd=16,
            n_layer=2,
            n_head=4,
            n_positions=32,
            n_ctx=32,
            bos_token_id=1,
            eos_token_id=2,
        )
    )
    gpt_rows = collect_for_model(
        gpt2,
        torch.randint(1, 31, (2, 12)),
        torch.tensor([[1] * 8 + [0] * 4, [1] * 12]),
        {
            "model": "local-random-gpt2",
            "dataset": "synthetic",
            "task": "smoke",
            "split": "none",
            "seq_len": 12,
            "model_type": "gpt2",
            "num_layers": 2,
            "hidden_dim": 16,
            "num_heads": 4,
            "norm_type": "LayerNorm",
            "attention_type": "causal",
        },
    )


    llama = LlamaModel(
        LlamaConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=32,
        )
    )
    llama_rows = collect_for_model(
        llama,
        torch.randint(1, 31, (2, 12)),
        torch.tensor([[1] * 9 + [0] * 3, [1] * 12]),
        {
            "model": "local-random-llama",
            "dataset": "synthetic",
            "task": "smoke",
            "split": "none",
            "seq_len": 12,
            "model_type": "llama",
            "num_layers": 2,
            "hidden_dim": 16,
            "num_heads": 4,
            "norm_type": "RMSNorm",
            "attention_type": "causal",
        },
    )

    rows = bert_rows + gpt_rows + llama_rows
    assert rows, "no rows collected"
    required = {"logZ", "r", "softmax_overhead", "effective_support_ratio"}
    missing = [col for col in required if any(row.get(col) is None for row in rows)]
    assert not missing, f"missing required values: {missing}"
    assert any(row.get("rms_denom") is not None for row in rows if row["model"] == "local-random-llama")
    assert any(row.get("ln1_denom") is not None for row in rows if row["model"] == "local-random-bert")
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(output, index=False)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
