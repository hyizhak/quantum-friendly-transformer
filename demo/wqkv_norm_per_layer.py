from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

# --- Model Configuration ---
# Uncomment and modify these if switching models
# model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
# safe_model_name = "llama_3_8b_4096"
# label_model_name = r'Llama-3-$8b$'

# model_name = "Qwen/Qwen2.5-3B-Instruct"
# safe_model_name = "qwen_2.5_3b_2048"
# label_model_name = r'Qwen2.5-$3b$'

# Using Mistral-Nemo as configured:
model_name = "mistralai/Mistral-Nemo-Base-2407"
safe_model_name = "mistral_nemo_12b_5120"
label_model_name = r'Mistral-Nemo-$12b$'

# --- Load the tokenizer and model ---
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
model.eval()

# --- Aggregate column norms across all layers ---
all_wq_norms = []
all_wk_norms = []
all_wv_norms = []

# Iterate over all layers of the model; assumes model.model.layers exists.
for i, layer in enumerate(model.model.layers):
    # Retrieve Q, K, V projection weights and move to CPU.
    Wq = layer.self_attn.q_proj.weight.detach().cpu()
    Wk = layer.self_attn.k_proj.weight.detach().cpu()
    Wv = layer.self_attn.v_proj.weight.detach().cpu()
    
    # Compute the ℓ₂ norm for each column (norm over rows).
    col_norms_Wq = Wq.norm(dim=0)
    col_norms_Wk = Wk.norm(dim=0)
    col_norms_Wv = Wv.norm(dim=0)
    
    # Append the computed norms to the corresponding list.
    all_wq_norms.append(col_norms_Wq)
    all_wk_norms.append(col_norms_Wk)
    all_wv_norms.append(col_norms_Wv)

# Concatenate all column norms from each layer into one tensor per projection.
all_wq_norms = torch.cat(all_wq_norms)
all_wk_norms = torch.cat(all_wk_norms)
all_wv_norms = torch.cat(all_wv_norms)

# Compute aggregated mean and variance for each projection.
Wq_mean = all_wq_norms.mean().item()
Wq_var  = all_wq_norms.var().item()
Wk_mean = all_wk_norms.mean().item()
Wk_var  = all_wk_norms.var().item()
Wv_mean = all_wv_norms.mean().item()
Wv_var  = all_wv_norms.var().item()

# Compute the overall (combined) statistics for Wq, Wk, and Wv.
all_combined = torch.cat([all_wq_norms, all_wk_norms, all_wv_norms])
total_mean = all_combined.mean().item()
total_var  = all_combined.var().item()

# Extract the d_hidden value from one of the layers (assumes all layers are identical in this regard)
d_hidden = model.model.layers[0].self_attn.q_proj.weight.shape[1]

# --- Save Aggregated Statistics to CSV ---
# We create one row CSV with the following columns:
# d_hidden, Wq_mean, Wq_var, Wk_mean, Wk_var, Wv_mean, Wv_var, Total_mean, Total_var
stats_dict = {
    "d_hidden": d_hidden,
    "Wq_mean": Wq_mean,
    "Wq_var": Wq_var,
    "Wk_mean": Wk_mean,
    "Wk_var": Wk_var,
    "Wv_mean": Wv_mean,
    "Wv_var": Wv_var,
    "Total_mean": total_mean,
    "Total_var": total_var,
}

df_stats = pd.DataFrame([stats_dict])
csv_filename = f"wqkv_norm_stats_aggregated_{safe_model_name}.csv"
df_stats.to_csv(csv_filename, index=False)
print(f"Saved aggregated norm statistics to {csv_filename}")

# --- Optional: Bar Plot of Aggregated Means with Error Bars ---
groups = ["Wq", "Wk", "Wv", "Total"]
means = [Wq_mean, Wk_mean, Wv_mean, total_mean]
variances = [Wq_var, Wk_var, Wv_var, total_var]

plt.figure(figsize=(8, 6))
plt.bar(groups, means, yerr=variances, capsize=5)
plt.xlabel("Projection Group")
plt.ylabel(r"Mean Column $\ell_2$ Norm")
plt.ylim(0, 1.3)
plt.title(rf"Aggregated Mean Column $\ell_2$ Norms for QKV Weights ({label_model_name})")
plt.tight_layout()
bar_plot_filename = f"wqkv_aggregated_bar_{safe_model_name}.pdf"
plt.savefig(bar_plot_filename)
plt.show()
print(f"Saved aggregated bar plot with error bars to {bar_plot_filename}")