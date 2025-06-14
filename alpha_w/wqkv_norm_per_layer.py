from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pandas as pd

def get_base_model(model):
    prefix = model.base_model_prefix
    return getattr(model, prefix)

def get_transformer_layers(base_model):
    if hasattr(base_model, 'layers'):
        return base_model.layers
    if hasattr(base_model, 'h'):
        return base_model.h
    if hasattr(base_model, 'encoder') and hasattr(base_model.encoder, 'layer'):
        return base_model.encoder.layer
    raise ValueError(f"Cannot find transformer layers on {base_model.__class__}")

def get_qkv_weights(layer):
    # 1) GPT-style / Llama / Mistral
    if hasattr(layer, 'self_attn'):
        attn = layer.self_attn
        return attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight

    # 2) BERT / RoBERTa
    if hasattr(layer, 'attention') and hasattr(layer.attention, 'self'):
        attn = layer.attention.self
        return attn.query.weight, attn.key.weight, attn.value.weight

    # 3) GPT-2 / GPT-Neo / GPT-J style
    #    – these pack Q,K,V into one Conv1D c_attn weight of shape [in, 3*out]
    if hasattr(layer, 'attn') and hasattr(layer.attn, 'c_attn'):
        W = layer.attn.c_attn.weight  # shape [hidden, 3*hidden]
        hidden = W.shape[1] // 3
        Wq = W[:, :hidden]
        Wk = W[:, hidden:2*hidden]
        Wv = W[:, 2*hidden:]
        return Wq, Wk, Wv

    raise ValueError(f"No self-attention block found in layer {layer.__class__.__name__}")

all_stats = []

for model_name in [
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "mistralai/Mistral-Nemo-Base-2407",
    'bert-base-uncased',
    'roberta-base',
    'distilgpt2',
    'gpt2',
    'gpt2-medium',
    'gpt2-large',
    'gpt2-xl',
    'openai-gpt',
    'meta-llama/Llama-2-7b-hf',
    'TinyLlama/Tinyllama-1.1B-chat-v1.0',
    'mistralai/Mistral-7B-v0.1',
]:
    safe_name = model_name.replace('/', '_').replace('-', '_')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    model.eval()

    base   = get_base_model(model)
    layers = get_transformer_layers(base)

    all_q, all_k, all_v = [], [], []
    for layer in layers:
        Wq, Wk, Wv = [w.detach().cpu() for w in get_qkv_weights(layer)]
        all_q.append(Wq.norm(dim=0))
        all_k.append(Wk.norm(dim=0))
        all_v.append(Wv.norm(dim=0))

    all_q = torch.cat(all_q)
    all_k = torch.cat(all_k)
    all_v = torch.cat(all_v)
    combined = torch.cat([all_q, all_k, all_v])

    # --- build stats dict ---
    stats_dict = {
        "model":    model_name,
        "d_hidden": all_q.shape[0] // len(layers),
        "Wq_mean":  all_q.mean().item(),
        "Wq_var":   all_q.var().item(),
        "Wk_mean":  all_k.mean().item(),
        "Wk_var":   all_k.var().item(),
        "Wv_mean":  all_v.mean().item(),
        "Wv_var":   all_v.var().item(),
        "total_mean": combined.mean().item(),
        "total_var":  combined.var().item(),
    }
    all_stats.append(stats_dict)

# --- save to CSV ---
df = pd.DataFrame(all_stats)
out_csv = f"./stats/qkv_stats.csv"
df.to_csv(out_csv, index=False)
print(f"Saved stats to {out_csv}")