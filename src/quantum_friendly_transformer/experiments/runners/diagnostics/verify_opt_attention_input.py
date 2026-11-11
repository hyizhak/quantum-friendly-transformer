"""Verify that the recorded OPT LayerNorm output is exactly what attention receives."""

import argparse
import csv
import gc
from pathlib import Path


DEFAULT_MODELS = [
    "facebook/opt-1.3b",
    "facebook/opt-2.7b",
    "facebook/opt-6.7b",
    "facebook/opt-30b",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument(
        "--output",
        default=(
            "experiments/outputs/pre_norm_opt_n_sweep/tables/"
            "attention_input_boundary_verification.csv"
        ),
    )
    return parser.parse_args()


def verify_model(torch, AutoModel, model_name, device, cache_dir):
    model = AutoModel.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        cache_dir=cache_dir,
        local_files_only=True,
        low_cpu_mem_usage=True,
        device_map={"": device},
    )
    model.eval()
    decoder = getattr(model, "decoder", None)
    if decoder is None:
        decoder = model.model.decoder
    layers = list(decoder.layers)
    if not model.config.do_layer_norm_before:
        raise AssertionError(f"{model_name} is not configured as pre-LayerNorm")

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

        handles.append(layer.self_attn_layer_norm.register_forward_hook(norm_hook))
        handles.append(
            layer.self_attn.register_forward_pre_hook(
                attention_pre_hook,
                with_kwargs=True,
            )
        )

    input_ids = torch.tensor(
        [[2, 5, 7, 11, 13, 17, 19, 23]],
        dtype=torch.long,
        device=device,
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

    if pending:
        raise AssertionError(f"unconsumed LayerNorm outputs at layers {sorted(pending)}")
    if len(comparisons) != len(layers):
        raise AssertionError(
            f"compared {len(comparisons)} attention inputs for {len(layers)} layers"
        )

    row = {
        "model": model_name,
        "revision": getattr(model.config, "_commit_hash", None),
        "hidden_dim": model.config.hidden_size,
        "num_layers": len(layers),
        "do_layer_norm_before": model.config.do_layer_norm_before,
        "layers_compared": len(comparisons),
        "all_exactly_equal": all(item["exactly_equal"] for item in comparisons),
        "all_same_storage": all(item["same_storage"] for item in comparisons),
        "maximum_absolute_difference": max(
            item["max_abs_difference"] for item in comparisons
        ),
    }
    del model, input_ids, attention_mask
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return row


def main():
    args = parse_args()
    import torch
    from transformers import AutoModel

    rows = []
    for model_name in args.models:
        row = verify_model(torch, AutoModel, model_name, args.device, args.cache_dir)
        rows.append(row)
        print(row, flush=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {output}", flush=True)


if __name__ == "__main__":
    main()
