from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

# Specify the model checkpoint for Llama-2-7b-chat-hf
# model_name = "meta-llama/Llama-2-7b-chat-hf"
# # Create a safe version of the model name for filenames
# safe_model_name = "llama_2_7b_4096"
# label_model_name = r'Llama-2-$7b$'

# Specify the model checkpoint for Qwen
model_name = "Qwen/Qwen2.5-3B-Instruct"
# Create a safe version of the model name for filenames
safe_model_name = "qwen_2.5_3b_2048"
label_model_name = r'Qwen2.5-$3b$'

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float16, trust_remote_code=True
)
model.eval()

# List to collect norm statistics for each layer
layers_stats = []

# Iterate over all layers in the model
for i, layer in enumerate(model.model.layers):
    # Access QKV weights for the current layer
    Wq = layer.self_attn.q_proj.weight.detach().cpu()
    Wk = layer.self_attn.k_proj.weight.detach().cpu()
    Wv = layer.self_attn.v_proj.weight.detach().cpu()

    # Compute column norms for each matrix
    col_norms_Wq = Wq.norm(dim=0)
    col_norms_Wk = Wk.norm(dim=0)
    col_norms_Wv = Wv.norm(dim=0)

    # Compute mean and variance for the column norms
    Wq_mean = col_norms_Wq.mean().item()
    Wq_var = col_norms_Wq.var().item()
    Wk_mean = col_norms_Wk.mean().item()
    Wk_var = col_norms_Wk.var().item()
    Wv_mean = col_norms_Wv.mean().item()
    Wv_var = col_norms_Wv.var().item()

    # Save the stats for the current layer
    layers_stats.append({
        "layer": i,
        "Wq_mean": Wq_mean,
        "Wq_var": Wq_var,
        "Wk_mean": Wk_mean,
        "Wk_var": Wk_var,
        "Wv_mean": Wv_mean,
        "Wv_var": Wv_var,
    })

# Create a DataFrame from the collected statistics and save to CSV
df_stats = pd.DataFrame(layers_stats)
csv_filename = f"wqkv_norm_stats_{safe_model_name}.csv"
df_stats.to_csv(csv_filename, index=False)
print(f"Saved norm statistics to {csv_filename}")

# Scatter plot for mean column norms with variance as error bars
plt.figure(figsize=(12, 6))
plt.errorbar(df_stats["layer"], df_stats["Wq_mean"], yerr=df_stats["Wq_var"],
             fmt='o', label=r"Wq $\ell_2$ Norm Mean")
plt.errorbar(df_stats["layer"], df_stats["Wk_mean"], yerr=df_stats["Wk_var"],
             fmt='o', label=r"Wk $\ell_2$ Norm Mean")
plt.errorbar(df_stats["layer"], df_stats["Wv_mean"], yerr=df_stats["Wv_var"],
             fmt='o', label=r"Wv $\ell_2$ Norm Mean")
plt.xlabel("Layer", fontsize=14)
plt.ylabel(r"Mean Column $\ell_2$ Norm", fontsize=14)
plt.title(rf"Mean Column $\ell_2$ Norms for QKV Weights per Layer ({label_model_name})", fontsize=16)
plt.legend(fontsize=12)
plt.tight_layout()
mean_plot_filename = f"wqkv_mean_scatter_{safe_model_name}.pdf"
plt.savefig(mean_plot_filename)
plt.show()
print(f"Saved mean scatter plot with error bars to {mean_plot_filename}")