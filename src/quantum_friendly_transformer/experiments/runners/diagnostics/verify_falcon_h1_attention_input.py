"""Verify the Falcon-H1 tensor recorded as the attention input at runtime."""

import argparse
import csv
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="tiiuae/Falcon-H1-34B-Base")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument(
        "--output",
        default=(
            "experiments/outputs/aggregate_deep_falcon_h1_34b/"
            "attention_input_boundary_verification.csv"
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    import torch
    from transformers import AutoModel

    model = AutoModel.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        cache_dir=args.cache_dir,
        local_files_only=True,
        low_cpu_mem_usage=True,
        device_map={"": args.device},
    )
    model.eval()
    layers = list(model.layers)
    multiplier = float(model.config.attention_in_multiplier)
    pending = {}
    comparisons = []
    handles = []

    for layer_index, layer in enumerate(layers, start=1):
        def norm_hook(_module, _inputs, output, index=layer_index):
            pending[index] = output

        def attention_pre_hook(_module, inputs, kwargs, index=layer_index):
            attention_input = kwargs.get("hidden_states")
            if attention_input is None:
                attention_input = inputs[0]
            norm_output = pending.pop(index)
            difference = (attention_input.float() - norm_output.float()).abs().max()
            comparisons.append(
                {
                    "layer": index,
                    "exactly_equal": bool(torch.equal(attention_input, norm_output)),
                    "same_storage": bool(
                        attention_input.untyped_storage().data_ptr()
                        == norm_output.untyped_storage().data_ptr()
                    ),
                    "max_abs_difference": float(difference.item()),
                }
            )

        handles.append(layer.input_layernorm.register_forward_hook(norm_hook))
        handles.append(
            layer.self_attn.register_forward_pre_hook(
                attention_pre_hook,
                with_kwargs=True,
            )
        )

    input_ids = torch.tensor(
        [[2, 5, 7, 11, 13, 17, 19, 23]],
        dtype=torch.long,
        device=args.device,
    )
    attention_mask = torch.ones_like(input_ids)
    try:
        with torch.inference_mode():
            model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
    finally:
        for handle in handles:
            handle.remove()

    if multiplier != 1.0:
        raise AssertionError(
            "The norm-output measurement is only the exact attention input when "
            f"attention_in_multiplier is 1.0; got {multiplier}."
        )
    if pending:
        raise AssertionError(f"unconsumed norm outputs at layers {sorted(pending)}")
    if len(comparisons) != len(layers):
        raise AssertionError(
            f"compared {len(comparisons)} attention inputs for {len(layers)} layers"
        )

    row = {
        "model": args.model,
        "revision": getattr(model.config, "_commit_hash", None),
        "hidden_dim": model.config.hidden_size,
        "num_layers": len(layers),
        "attention_in_multiplier": multiplier,
        "layers_compared": len(comparisons),
        "all_exactly_equal": all(item["exactly_equal"] for item in comparisons),
        "all_same_storage": all(item["same_storage"] for item in comparisons),
        "maximum_absolute_difference": max(
            item["max_abs_difference"] for item in comparisons
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    print(row, flush=True)
    print(f"wrote {output}", flush=True)


if __name__ == "__main__":
    main()
