import argparse
import math
import os
from collections import defaultdict
from pathlib import Path

from quantum_friendly_transformer.experiments.lib.common import (
    ensure_parent,
    finite_or_none,
    glue_split_name,
    model_config_summary,
    metadata_path,
    now_metadata,
    package_versions,
    seed_everything,
    write_json,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Collect NLP forward-pass diagnostics.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True, choices=["glue", "wikitext", "pile_arxiv", "pile_github"])
    parser.add_argument("--task", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--seq-lens", type=int, nargs="+", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--sample-rows-per-layer-head", type=int, default=4096)
    parser.add_argument("--spectral-iters", type=int, default=8)
    parser.add_argument("--z-only", action="store_true", help="Collect Z/N and norm-denominator diagnostics without entropy/effective-support tensors.")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def pick_device(torch, requested):
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def torch_dtype(torch, name):
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def glue_columns(task):
    columns = {
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
    if task not in columns:
        raise ValueError(f"unsupported GLUE task: {task}")
    return columns[task]


def make_batches(torch, records, batch_size):
    for start in range(0, len(records), batch_size):
        chunk = records[start : start + batch_size]
        yield {
            "input_ids": torch.tensor([x["input_ids"] for x in chunk], dtype=torch.long),
            "attention_mask": torch.tensor([x["attention_mask"] for x in chunk], dtype=torch.long),
            "sample_index": [x["sample_index"] for x in chunk],
        }


def build_glue_records(load_dataset, tokenizer, task, split, seq_len, max_samples, cache_dir):
    dataset = load_dataset("glue", task, split=glue_split_name(task, split), cache_dir=cache_dir)
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    col1, col2 = glue_columns(task)
    records = []
    for idx, example in enumerate(dataset):
        if col2 is None:
            encoded = tokenizer(
                example[col1],
                truncation=True,
                padding="max_length",
                max_length=seq_len,
            )
        else:
            encoded = tokenizer(
                example[col1],
                example[col2],
                truncation=True,
                padding="max_length",
                max_length=seq_len,
            )
        records.append(
            {
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
                "sample_index": idx,
            }
        )
    return records


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


def build_wikitext_records(load_dataset, tokenizer, config, split, seq_len, max_samples, max_tokens, cache_dir):
    dataset = load_dataset("wikitext", config or "wikitext-103-raw-v1", split=split, cache_dir=cache_dir)
    text = wikitext_prefix(dataset, max_tokens)
    tokenize_kwargs = {"add_special_tokens": False}
    if max_tokens is not None:
        tokenize_kwargs.update({"truncation": True, "max_length": max_tokens})
    encoded = tokenizer(text, **tokenize_kwargs)
    ids = encoded["input_ids"]
    if max_tokens is not None:
        ids = ids[:max_tokens]
    records = []
    for sample_index, start in enumerate(range(0, len(ids) - seq_len + 1, seq_len)):
        if max_samples is not None and sample_index >= max_samples:
            break
        records.append(
            {
                "input_ids": ids[start : start + seq_len],
                "attention_mask": [1] * seq_len,
                "sample_index": sample_index,
            }
        )
    if not records:
        raise ValueError(f"no full WikiText chunks for seq_len={seq_len}")
    return records


def build_pile_records(load_dataset, tokenizer, dataset_name, split, seq_len, max_samples, max_tokens, cache_dir):
    dataset = load_dataset(dataset_name, split=split, streaming=True, cache_dir=cache_dir)
    text = wikitext_prefix(dataset, max_tokens)
    tokenize_kwargs = {"add_special_tokens": False}
    if max_tokens is not None:
        tokenize_kwargs.update({"truncation": True, "max_length": max_tokens})
    encoded = tokenizer(text, **tokenize_kwargs)
    ids = encoded["input_ids"]
    if max_tokens is not None:
        ids = ids[:max_tokens]
    records = []
    for sample_index, start in enumerate(range(0, len(ids) - seq_len + 1, seq_len)):
        if max_samples is not None and sample_index >= max_samples:
            break
        records.append(
            {
                "input_ids": ids[start : start + seq_len],
                "attention_mask": [1] * seq_len,
                "sample_index": sample_index,
            }
        )
    if not records:
        raise ValueError(f"no full {dataset_name} chunks for seq_len={seq_len}")
    return records


class ForwardStatHooks:
    def __init__(self, model, torch):
        self.model = model
        self.torch = torch
        self.handles = []
        self.attn_inputs = {}
        self.norm_denoms = defaultdict(dict)
        config_model_type = getattr(model.config, "model_type", "unknown")
        self.model_type = "gemma3_text" if config_model_type == "gemma3" and hasattr(model, "language_model") else config_model_type
        self.layers = []
        self.layer_windows = {}
        self._register()

    def clear(self):
        self.attn_inputs.clear()
        self.norm_denoms.clear()

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _register_norm(self, module, layer_index, slot, rms=False):
        torch = self.torch

        def hook(mod, inputs):
            x = inputs[0].detach().float()
            eps = getattr(mod, "variance_epsilon", None) or getattr(mod, "eps", 1e-5)
            if rms:
                denom = torch.sqrt(torch.mean(x * x, dim=-1) + eps)
            else:
                centered = x - x.mean(dim=-1, keepdim=True)
                denom = torch.sqrt(torch.mean(centered * centered, dim=-1) + eps)
            self.norm_denoms[layer_index][slot] = denom

        self.handles.append(module.register_forward_pre_hook(hook))

    def _detach_tree(self, value):
        if hasattr(value, "detach"):
            return value.detach()
        if isinstance(value, tuple):
            return tuple(self._detach_tree(item) for item in value)
        if isinstance(value, list):
            return [self._detach_tree(item) for item in value]
        if isinstance(value, dict):
            return {key: self._detach_tree(item) for key, item in value.items()}
        return value

    def _register_attn(self, module, layer_index):
        def hook(mod, inputs, kwargs, output):
            self.attn_inputs[layer_index] = {
                "args": self._detach_tree(inputs),
                "kwargs": self._detach_tree(kwargs),
            }

        self.layers.append((layer_index, module))
        self.handles.append(module.register_forward_hook(hook, with_kwargs=True))

    def _register(self):
        if self.model_type in {"bert", "roberta"}:
            base = getattr(self.model, self.model_type, self.model)
            for idx, layer in enumerate(base.encoder.layer):
                self._register_attn(layer.attention.self, idx)
                self._register_norm(layer.attention.output.LayerNorm, idx, "ln1")
                self._register_norm(layer.output.LayerNorm, idx, "ln2")
            return

        if self.model_type == "gpt2":
            base = getattr(self.model, "transformer", self.model)
            for idx, block in enumerate(base.h):
                self._register_attn(block.attn, idx)
                self._register_norm(block.ln_1, idx, "ln1")
                self._register_norm(block.ln_2, idx, "ln2")
            return

        if self.model_type == "gpt_neo":
            base = getattr(self.model, "transformer", self.model)
            attention_layers = getattr(self.model.config, "attention_layers", [])
            window_size = getattr(self.model.config, "window_size", None)
            for idx, block in enumerate(base.h):
                self._register_attn(block.attn.attention, idx)
                if idx < len(attention_layers) and attention_layers[idx] == "local":
                    self.layer_windows[idx] = window_size
                self._register_norm(block.ln_1, idx, "ln1")
                self._register_norm(block.ln_2, idx, "ln2")
            return

        if self.model_type == "gpt_neox":
            base = getattr(self.model, "gpt_neox", self.model)
            for idx, layer in enumerate(base.layers):
                self._register_attn(layer.attention, idx)
                self._register_norm(layer.input_layernorm, idx, "ln1")
                self._register_norm(layer.post_attention_layernorm, idx, "ln2")
            return

        if self.model_type == "opt":
            base = getattr(self.model, "decoder", getattr(self.model, "model", self.model))
            for idx, layer in enumerate(base.layers):
                self._register_attn(layer.self_attn, idx)
                self._register_norm(layer.self_attn_layer_norm, idx, "ln1")
                self._register_norm(layer.final_layer_norm, idx, "ln2")
            return

        if self.model_type == "bloom":
            base = getattr(self.model, "transformer", self.model)
            for idx, block in enumerate(base.h):
                self._register_attn(block.self_attention, idx)
                self._register_norm(block.input_layernorm, idx, "ln1")
                self._register_norm(block.post_attention_layernorm, idx, "ln2")
            return

        if self.model_type in {"llama", "mistral", "qwen2", "qwen3", "olmo", "phi3", "gemma", "gemma2", "gemma3_text", "smollm3"}:
            if self.model_type == "gemma3_text" and hasattr(self.model, "language_model"):
                base = self.model.language_model
            else:
                base = getattr(self.model, "model", self.model)
            for idx, layer in enumerate(base.layers):
                self._register_attn(layer.self_attn, idx)
                window = getattr(layer.self_attn, "sliding_window", None)
                if window is None and self.model_type == "mistral":
                    window = getattr(self.model.config, "sliding_window", None)
                if window is not None:
                    self.layer_windows[idx] = window
                self._register_norm(layer.input_layernorm, idx, "rms", rms=self.model_type in {"llama", "mistral", "qwen2", "qwen3", "phi3", "gemma", "gemma2", "gemma3_text", "smollm3"})
                self._register_norm(layer.post_attention_layernorm, idx, "rms2", rms=self.model_type in {"llama", "mistral", "qwen2", "qwen3", "phi3", "gemma", "gemma2", "gemma3_text", "smollm3"})
            return

        if self.model_type == "phi":
            base = getattr(self.model, "model", self.model)
            for idx, layer in enumerate(base.layers):
                self._register_attn(layer.self_attn, idx)
                self._register_norm(layer.input_layernorm, idx, "ln1")
            return

        raise ValueError(
            "attention-logit reconstruction is implemented for bert, roberta, gpt2, gpt_neo, gpt_neox, "
            "opt, bloom, llama, mistral, qwen2, qwen3, olmo, phi, phi3, gemma, gemma2, gemma3_text, and smollm3; "
            f"got {self.model_type}"
        )

    def attention_scores(self, layer_index, module, query_positions=None):
        torch = self.torch
        attn_record = self.attn_inputs[layer_index]
        if isinstance(attn_record, dict) and "args" in attn_record:
            attn_args = attn_record.get("args") or ()
            attn_kwargs = attn_record.get("kwargs") or {}
        else:
            attn_args = attn_record if isinstance(attn_record, tuple) else (attn_record,)
            attn_kwargs = {}
        x = attn_args[0] if len(attn_args) else attn_kwargs.get("hidden_states")
        if x is None:
            raise RuntimeError(f"missing hidden_states for layer {layer_index}")
        if self.model_type in {"bert", "roberta"}:
            query = module.query(x)
            key = module.key(x)
            num_heads = module.num_attention_heads
            head_dim = module.attention_head_size
            query = query.view(query.shape[0], query.shape[1], num_heads, head_dim).permute(0, 2, 1, 3)
            key = key.view(key.shape[0], key.shape[1], num_heads, head_dim).permute(0, 2, 1, 3)
            return torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(head_dim)

        if self.model_type == "gpt2":
            qkv = module.c_attn(x)
            hidden = qkv.shape[-1] // 3
            query, key, _ = qkv.split(hidden, dim=2)
            num_heads = module.num_heads
            head_dim = hidden // num_heads
            query = query.view(query.shape[0], query.shape[1], num_heads, head_dim).permute(0, 2, 1, 3)
            key = key.view(key.shape[0], key.shape[1], num_heads, head_dim).permute(0, 2, 1, 3)
            scores = torch.matmul(query, key.transpose(-1, -2))
            if getattr(module, "scale_attn_weights", True):
                scores = scores / math.sqrt(head_dim)
            if getattr(module, "scale_attn_by_inverse_layer_idx", False):
                layer_idx = getattr(module, "layer_idx", layer_index)
                scores = scores / float(layer_idx + 1)
            return scores

        if self.model_type == "gpt_neo":
            query = module.q_proj(x)
            key = module.k_proj(x)
            query = module._split_heads(query, module.num_heads, module.head_dim)
            key = module._split_heads(key, module.num_heads, module.head_dim)
            return torch.matmul(query.float(), key.float().transpose(-1, -2))

        if self.model_type == "gpt_neox":
            from transformers.models.gpt_neox.modeling_gpt_neox import apply_rotary_pos_emb

            position_embeddings = attn_args[1] if len(attn_args) > 1 else attn_kwargs.get("position_embeddings")
            if position_embeddings is None:
                raise RuntimeError(f"missing position_embeddings for {self.model_type} layer {layer_index}")
            input_shape = x.shape[:-1]
            hidden_shape = (*input_shape, -1, 3 * module.head_size)
            qkv = module.query_key_value(x).view(hidden_shape).transpose(1, 2)
            query, key, _ = qkv.chunk(3, dim=-1)
            cos, sin = position_embeddings
            query, key = apply_rotary_pos_emb(query, key, cos, sin)
            return torch.matmul(query, key.transpose(-1, -2)) * module.scaling

        if self.model_type == "opt":
            batch, _, _ = x.shape
            query = module.q_proj(x) * module.scaling
            key = module.k_proj(x)
            query = query.view(batch, -1, module.num_heads, module.head_dim).transpose(1, 2)
            key = key.view(batch, -1, module.num_heads, module.head_dim).transpose(1, 2)
            return torch.matmul(query, key.transpose(-1, -2))

        if self.model_type == "bloom":
            alibi = attn_kwargs.get("alibi")
            if alibi is None and len(attn_args) > 2:
                alibi = attn_args[2]
            if alibi is None:
                raise RuntimeError(f"missing alibi for bloom layer {layer_index}")
            batch_size, q_length, _ = x.shape
            query, key, _ = module._reshape(module.query_key_value(x))
            query = query.reshape(batch_size * module.num_heads, -1, module.head_dim)
            key = key.reshape(batch_size * module.num_heads, -1, module.head_dim).transpose(-1, -2)
            scores = alibi.baddbmm(batch1=query, batch2=key, beta=module.beta, alpha=module.inv_norm_factor)
            return scores.view(batch_size, module.num_heads, q_length, -1)

        if self.model_type in {"llama", "mistral", "qwen2", "qwen3", "olmo", "gemma", "gemma2", "smollm3"}:
            if self.model_type == "qwen2":
                from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv
            elif self.model_type == "qwen3":
                from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, repeat_kv
            elif self.model_type == "olmo":
                from transformers.models.olmo.modeling_olmo import apply_rotary_pos_emb, repeat_kv
            elif self.model_type == "gemma":
                from transformers.models.gemma.modeling_gemma import apply_rotary_pos_emb, repeat_kv
            elif self.model_type == "gemma2":
                from transformers.models.gemma2.modeling_gemma2 import apply_rotary_pos_emb, repeat_kv
            elif self.model_type == "smollm3":
                from transformers.models.smollm3.modeling_smollm3 import apply_rotary_pos_emb, repeat_kv
            else:
                from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

            position_embeddings = attn_args[1] if len(attn_args) > 1 else attn_kwargs.get("position_embeddings")
            if position_embeddings is None:
                raise RuntimeError(f"missing position_embeddings for {self.model_type} layer {layer_index}")
            input_shape = x.shape[:-1]
            hidden_shape = (*input_shape, -1, module.head_dim)
            query = module.q_proj(x).view(hidden_shape)
            key = module.k_proj(x).view(hidden_shape)
            if self.model_type == "qwen3":
                query = module.q_norm(query)
                key = module.k_norm(key)
            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            if self.model_type == "olmo" and getattr(module.config, "clip_qkv", None) is not None:
                clip = module.config.clip_qkv
                query = query.clamp(min=-clip, max=clip)
                key = key.clamp(min=-clip, max=clip)
            cos, sin = position_embeddings
            if self.model_type != "smollm3" or getattr(module, "use_rope", True):
                query, key = apply_rotary_pos_emb(query, key, cos, sin)
            key = repeat_kv(key, module.num_key_value_groups)
            if query_positions is not None:
                query = query.index_select(-2, query_positions)
            scaling = getattr(module, "scaling", 1.0 / math.sqrt(module.head_dim))
            scores = torch.matmul(query, key.transpose(-1, -2)) * scaling
            if self.model_type == "gemma2":
                softcap = getattr(module, "attn_logit_softcapping", None)
                if softcap is not None:
                    scores = torch.tanh(scores / softcap) * softcap
            return scores

        if self.model_type == "phi":
            from transformers.models.phi.modeling_phi import apply_rotary_pos_emb, repeat_kv

            position_embeddings = attn_args[1] if len(attn_args) > 1 else attn_kwargs.get("position_embeddings")
            if position_embeddings is None:
                raise RuntimeError(f"missing position_embeddings for phi layer {layer_index}")
            input_shape = x.shape[:-1]
            hidden_shape = (*input_shape, -1, module.head_dim)
            query = module.q_proj(x).view(hidden_shape).transpose(1, 2)
            key = module.k_proj(x).view(hidden_shape).transpose(1, 2)
            if module.qk_layernorm:
                query = module.q_layernorm(query)
                key = module.k_layernorm(key)
            query_rot, query_pass = query[..., : module.rotary_ndims], query[..., module.rotary_ndims :]
            key_rot, key_pass = key[..., : module.rotary_ndims], key[..., module.rotary_ndims :]
            cos, sin = position_embeddings
            query_rot, key_rot = apply_rotary_pos_emb(query_rot, key_rot, cos, sin)
            query = torch.cat((query_rot, query_pass), dim=-1)
            key = torch.cat((key_rot, key_pass), dim=-1)
            key = repeat_kv(key, module.num_key_value_groups)
            return torch.matmul(query, key.transpose(-1, -2)) * module.scaling

        if self.model_type == "phi3":
            from transformers.models.phi3.modeling_phi3 import apply_rotary_pos_emb, repeat_kv

            position_embeddings = attn_args[1] if len(attn_args) > 1 else attn_kwargs.get("position_embeddings")
            if position_embeddings is None:
                raise RuntimeError(f"missing position_embeddings for phi3 layer {layer_index}")
            input_shape = x.shape[:-1]
            hidden_shape = (*input_shape, -1, module.head_dim)
            qkv = module.qkv_proj(x)
            query_pos = module.config.num_attention_heads * module.head_dim
            kv_size = module.num_key_value_heads * module.head_dim
            query = qkv[..., :query_pos].view(hidden_shape).transpose(1, 2)
            key = qkv[..., query_pos : query_pos + kv_size].view(hidden_shape).transpose(1, 2)
            cos, sin = position_embeddings
            query, key = apply_rotary_pos_emb(query, key, cos, sin)
            key = repeat_kv(key, module.num_key_value_groups)
            return torch.matmul(query, key.transpose(-1, -2)) * module.scaling

        if self.model_type == "gemma3_text":
            from transformers.models.gemma3.modeling_gemma3 import apply_rotary_pos_emb, repeat_kv

            position_embeddings = attn_args[1] if len(attn_args) > 1 else attn_kwargs.get("position_embeddings")
            if position_embeddings is None:
                raise RuntimeError(f"missing position_embeddings for gemma3_text layer {layer_index}")
            input_shape = x.shape[:-1]
            hidden_shape = (*input_shape, -1, module.head_dim)
            query = module.q_proj(x).view(hidden_shape).transpose(1, 2)
            key = module.k_proj(x).view(hidden_shape).transpose(1, 2)
            query = module.q_norm(query)
            key = module.k_norm(key)
            cos, sin = position_embeddings
            query, key = apply_rotary_pos_emb(query, key, cos, sin)
            key = repeat_kv(key, module.num_key_value_groups)
            scores = torch.matmul(query, key.transpose(-1, -2)) * module.scaling
            softcap = getattr(module, "attn_logit_softcapping", None)
            if softcap is not None:
                scores = torch.tanh(scores / softcap) * softcap
            return scores

        raise AssertionError("unreachable")


def estimate_spectral_norm(torch, matrix, iters):
    if matrix.numel() == 0:
        return float("nan")
    mat = matrix.float()
    v = torch.randn(mat.shape[1], device=mat.device, dtype=mat.dtype)
    v = v / (torch.linalg.vector_norm(v) + 1e-12)
    for _ in range(max(1, iters)):
        u = mat @ v
        u = u / (torch.linalg.vector_norm(u) + 1e-12)
        v = mat.transpose(0, 1) @ u
        v = v / (torch.linalg.vector_norm(v) + 1e-12)
    return float(torch.dot(u, mat @ v).abs().item())


def hidden_metrics_by_layer(torch, hidden_states, attention_mask, spectral_iters):
    valid = attention_mask.bool()
    metrics = {}
    # hidden_states[0] is the embedding/input state. Layer i output is i + 1.
    for layer_index, hidden in enumerate(hidden_states[1:]):
        hidden = hidden.detach()
        per_example = []
        for batch_index in range(hidden.shape[0]):
            rows = hidden[batch_index][valid[batch_index]]
            if rows.numel() == 0:
                per_example.append({})
                continue
            row_norms = torch.linalg.vector_norm(rows.float(), dim=-1)
            n_valid = max(1, rows.shape[0])
            spectral = estimate_spectral_norm(torch, rows, spectral_iters)
            per_example.append(
                {
                    "hidden_fro_over_sqrtN": float(torch.linalg.vector_norm(rows.float()).item() / math.sqrt(n_valid)),
                    "hidden_spectral_over_sqrtN": float(spectral / math.sqrt(n_valid)),
                    "hidden_max_row_norm": float(row_norms.max().item()),
                    "hidden_p95_row_norm": float(torch.quantile(row_norms, 0.95).item()),
                    "hidden_median_row_norm": float(torch.quantile(row_norms, 0.50).item()),
                }
            )
        metrics[layer_index] = per_example
    return metrics


def valid_attention_mask(torch, attention_mask, causal, sliding_window=None):
    key_valid = attention_mask.bool()
    batch, seq_len = key_valid.shape
    valid = key_valid[:, None, None, :].expand(batch, 1, seq_len, seq_len)
    pos = torch.arange(seq_len, device=attention_mask.device)
    if causal:
        causal_mask = pos[None, :] <= pos[:, None]
        valid = valid & causal_mask[None, None, :, :]
    if sliding_window is not None:
        # GPT-Neo/Gemma-style local attention keeps at most `sliding_window` causal keys per query.
        local_mask = (pos[None, :] > (pos[:, None] - int(sliding_window))) & (pos[None, :] <= pos[:, None])
        valid = valid & local_mask[None, None, :, :]
    query_valid = key_valid
    valid_counts = valid.sum(dim=-1).squeeze(1)
    return valid, query_valid, valid_counts


def norm_denom_at(denom, batch_index, token_pos, seq_len):
    if denom is None:
        return None
    if denom.ndim >= 2:
        return finite_or_none(denom[batch_index, token_pos].item())
    if denom.ndim == 1:
        flat_index = batch_index * seq_len + token_pos
        if denom.shape[0] == seq_len:
            return finite_or_none(denom[token_pos].item())
        if flat_index < denom.shape[0]:
            return finite_or_none(denom[flat_index].item())
        return None
    return finite_or_none(denom.item())


def append_attention_rows(
    torch,
    rows,
    sample_counts,
    hooks,
    batch,
    outputs,
    context,
    sample_rows_per_layer_head,
    spectral_iters,
    z_only=False,
):
    attention_mask = batch["attention_mask"]
    hidden_metrics = {} if z_only else hidden_metrics_by_layer(torch, outputs.hidden_states, attention_mask, spectral_iters)
    causal = hooks.model_type in {"gpt2", "gpt_neo", "gpt_neox", "opt", "bloom", "llama", "mistral", "qwen2", "qwen3", "olmo", "phi", "phi3", "gemma", "gemma2", "gemma3_text", "smollm3"}

    for layer_index, module in hooks.layers:
        valid_mask, query_valid, valid_counts = valid_attention_mask(
            torch, attention_mask, causal, hooks.layer_windows.get(layer_index)
        )
        valid_mask = valid_mask.to(attention_mask.device)
        query_valid = query_valid.to(attention_mask.device)
        valid_counts = valid_counts.to(attention_mask.device)
        full_seq_len = attention_mask.shape[-1]
        query_positions = None
        sampled_model_types = {"llama", "mistral", "qwen2", "qwen3", "olmo", "gemma", "gemma2", "smollm3"}
        if z_only and hooks.model_type in sampled_model_types:
            remaining_by_head = [
                sample_rows_per_layer_head
                - sample_counts[(context["model"], context["dataset"], context["task"], context["seq_len"], layer_index, head_index)]
                for head_index in range(module.config.num_attention_heads)
            ]
            max_remaining = max(remaining_by_head, default=0)
            eligible = query_valid.any(dim=0).nonzero(as_tuple=False).flatten()
            if eligible.numel() > max_remaining:
                perm = torch.randperm(eligible.numel(), device=eligible.device)[:max_remaining]
                eligible = eligible[perm]
            query_positions = eligible
            valid_mask = valid_mask.index_select(-2, query_positions)
            query_valid = query_valid.index_select(-1, query_positions)
            valid_counts = valid_counts.index_select(-1, query_positions)

        scores = hooks.attention_scores(layer_index, module, query_positions=query_positions).float()
        batch_size, num_heads, query_len, key_len = scores.shape
        expanded_valid = valid_mask.expand(batch_size, num_heads, query_len, key_len)
        scores = scores.masked_fill(~expanded_valid, float("-inf"))

        logz = torch.logsumexp(scores, dim=-1)
        counts = valid_counts[:, None, :].expand(batch_size, num_heads, query_len).float()
        log_r = logz - torch.log(counts.clamp_min(1.0))
        if z_only:
            entropy = None
            effective_support_ratio = None
            top1_mass = (scores.max(dim=-1).values - logz).exp()
        else:
            probs = torch.softmax(scores, dim=-1)
            log_probs = torch.log_softmax(scores, dim=-1)
            p_log_p = torch.where(expanded_valid, probs * log_probs, torch.zeros_like(probs))
            entropy = -p_log_p.sum(dim=-1)
            effective_support_ratio = entropy.exp() / counts.clamp_min(1.0)
            top1_mass = probs.max(dim=-1).values
        overhead = (-0.5 * log_r).exp()
        ratio = log_r.exp()

        layer_norms = hooks.norm_denoms.get(layer_index, {})
        ln1 = layer_norms.get("ln1")
        ln2 = layer_norms.get("ln2")
        rms = layer_norms.get("rms")
        rms2 = layer_norms.get("rms2")
        layer_hidden = hidden_metrics.get(layer_index, [])

        for head_index in range(num_heads):
            sample_key = (context["model"], context["dataset"], context["task"], context["seq_len"], layer_index, head_index)
            remaining = sample_rows_per_layer_head - sample_counts[sample_key]
            if remaining <= 0:
                continue
            candidate = (
                query_valid
                & torch.isfinite(logz[:, head_index, :])
                & (valid_counts > 0)
            ).nonzero(as_tuple=False)
            if candidate.numel() == 0:
                continue
            if candidate.shape[0] > remaining:
                perm = torch.randperm(candidate.shape[0], device=candidate.device)[:remaining]
                candidate = candidate[perm]
            sample_counts[sample_key] += int(candidate.shape[0])
            for batch_index_t, token_pos_t in candidate:
                batch_index = int(batch_index_t.item())
                query_index = int(token_pos_t.item())
                token_pos = (
                    int(query_positions[query_index].item())
                    if query_positions is not None
                    else query_index
                )
                hidden = layer_hidden[batch_index] if batch_index < len(layer_hidden) else {}
                rows.append(
                    {
                        **context,
                        "sample_index": int(batch["sample_index"][batch_index]),
                        "layer": layer_index,
                        "head": head_index,
                        "token_position": token_pos,
                        "valid_key_count": int(valid_counts[batch_index, query_index].item()),
                        "logZ": finite_or_none(logz[batch_index, head_index, query_index].item()),
                        "log_r": finite_or_none(log_r[batch_index, head_index, query_index].item()),
                        "r": finite_or_none(ratio[batch_index, head_index, query_index].item()),
                        "softmax_overhead": finite_or_none(overhead[batch_index, head_index, query_index].item()),
                        "entropy": finite_or_none(entropy[batch_index, head_index, query_index].item()) if entropy is not None else None,
                        "effective_support_ratio": finite_or_none(
                            effective_support_ratio[batch_index, head_index, query_index].item()
                        ) if effective_support_ratio is not None else None,
                        "top1_mass": finite_or_none(top1_mass[batch_index, head_index, query_index].item()),
                        "ln1_denom": norm_denom_at(ln1, batch_index, token_pos, full_seq_len),
                        "ln2_denom": norm_denom_at(ln2, batch_index, token_pos, full_seq_len),
                        "rms_denom": norm_denom_at(rms, batch_index, token_pos, full_seq_len),
                        "rms2_denom": norm_denom_at(rms2, batch_index, token_pos, full_seq_len),
                        **hidden,
                    }
                )


def main():
    args = parse_args()
    import pandas as pd
    import torch
    from datasets import load_dataset
    from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer

    seed_everything(args.seed)
    device = pick_device(torch, args.device)
    dtype = torch_dtype(torch, args.dtype)

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
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    config = AutoConfig.from_pretrained(
        args.model,
        cache_dir=args.cache_dir,
        local_files_only=args.offline,
        trust_remote_code=True,
    )
    model_loader = AutoModelForCausalLM if getattr(config, "model_type", None) in {"gemma3", "gemma3_text"} else AutoModel
    model = model_loader.from_pretrained(
        args.model,
        torch_dtype=dtype,
        cache_dir=args.cache_dir,
        local_files_only=args.offline,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()
    config_summary = model_config_summary(model)
    hooks = ForwardStatHooks(model, torch)
    rows = []
    sample_counts = defaultdict(int)

    try:
        with torch.no_grad():
            for seq_len in args.seq_lens:
                if args.dataset == "glue":
                    if not args.task:
                        raise ValueError("--task is required for GLUE")
                    records = build_glue_records(
                        load_dataset, tokenizer, args.task, args.split, seq_len, args.max_samples, args.cache_dir
                    )
                elif args.dataset == "wikitext":
                    records = build_wikitext_records(
                        load_dataset,
                        tokenizer,
                        args.config,
                        args.split,
                        seq_len,
                        args.max_samples,
                        args.max_tokens,
                        args.cache_dir,
                    )
                else:
                    pile_dataset_names = {
                        "pile_arxiv": "suolyer/pile_arxiv",
                        "pile_github": "suolyer/pile_github",
                    }
                    records = build_pile_records(
                        load_dataset,
                        tokenizer,
                        pile_dataset_names[args.dataset],
                        args.split,
                        seq_len,
                        args.max_samples,
                        args.max_tokens,
                        args.cache_dir,
                    )

                for batch in make_batches(torch, records, args.batch_size):
                    hooks.clear()
                    inputs = {
                        "input_ids": batch["input_ids"].to(device),
                        "attention_mask": batch["attention_mask"].to(device),
                    }
                    outputs = model(
                        **inputs,
                        output_hidden_states=not args.z_only,
                        return_dict=True,
                    )
                    batch_on_device = {
                        "attention_mask": inputs["attention_mask"],
                        "sample_index": batch["sample_index"],
                    }
                    context = {
                        "model": args.model,
                        "dataset": args.dataset,
                        "task": args.task or args.config or "",
                        "split": args.split,
                        "seq_len": seq_len,
                        **config_summary,
                    }
                    append_attention_rows(
                        torch,
                        rows,
                        sample_counts,
                        hooks,
                        batch_on_device,
                        outputs,
                        context,
                        args.sample_rows_per_layer_head,
                        args.spectral_iters,
                        z_only=args.z_only,
                    )
    finally:
        hooks.close()

    if not rows:
        raise RuntimeError("no diagnostic rows were collected")

    output = ensure_parent(args.output)
    pd.DataFrame(rows).to_parquet(output, index=False)
    write_json(
        metadata_path(output),
        {
            **now_metadata(),
            "model": args.model,
            "model_revision": getattr(model.config, "_commit_hash", None),
            "dataset": args.dataset,
            "task": args.task,
            "config": args.config,
            "split": args.split,
            "num_examples": args.max_samples,
            "max_tokens": args.max_tokens,
            "sequence_lengths": args.seq_lens,
            "batch_size": args.batch_size,
            "dtype": args.dtype,
            "seed": args.seed,
            "device": str(device),
            "cache_dir": args.cache_dir,
            "offline": args.offline,
            "output": str(output),
            "sample_rows_per_layer_head": args.sample_rows_per_layer_head,
            "z_only": args.z_only,
            "package_versions": package_versions(),
            **config_summary,
        },
    )
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
