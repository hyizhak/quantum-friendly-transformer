from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import torch
import matplotlib.pyplot as plt
import numpy as np
import os

cache_dir = "/home/users/nus/e1310988/scratch/huggingface"

os.environ['HF_HOME'] = cache_dir
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'

# Specify the model checkpoint for Llama2
# model_name = "meta-llama/Llama-2-7b-chat-hf"
model_name = f"{cache_dir}/hub/Llama-2-7b-chat-hf"

# # Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

model.eval()

print(model)

# Access QKV weights
Wq = model.model.layers[0].self_attn.q_proj.weight.detach().cpu()
Wk = model.model.layers[0].self_attn.k_proj.weight.detach().cpu()
Wv = model.model.layers[0].self_attn.v_proj.weight.detach().cpu()

# Compute row/column L2 norms for each matrix
row_norms_Wq = Wq.norm(dim=1)  # L2 norm of each row of Wq
col_norms_Wq = Wq.norm(dim=0)  # L2 norm of each column of Wq

row_norms_Wk = Wk.norm(dim=1)
col_norms_Wk = Wk.norm(dim=0)

row_norms_Wv = Wv.norm(dim=1)
col_norms_Wv = Wv.norm(dim=0)

# Create a figure with multiple subplots for histograms
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(16, 10))

num_bins = 30

# Helper to plot each histogram in a subplot
def plot_hist(ax, data, title):
    counts, bin_edges = np.histogram(data.cpu().numpy(), bins=num_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    ax.bar(bin_centers, counts, width=bin_edges[1] - bin_edges[0], edgecolor='black')
    ax.set_title(title)
    ax.set_xlim(0, 3)
    ax.set_xlabel("L2 Norm")
    ax.set_ylabel("Frequency")

# First row: row-norm histograms
plot_hist(axes[0, 0], row_norms_Wq, "Row L2 Norm - Wq")
plot_hist(axes[0, 1], row_norms_Wk, "Row L2 Norm - Wk")
plot_hist(axes[0, 2], row_norms_Wv, "Row L2 Norm - Wv")

# Second row: column-norm histograms
plot_hist(axes[1, 0], col_norms_Wq, "Column L2 Norm - Wq")
plot_hist(axes[1, 1], col_norms_Wk, "Column L2 Norm - Wk")
plot_hist(axes[1, 2], col_norms_Wv, "Column L2 Norm - Wv")

plt.suptitle("Histograms of Row/Column L2 Norms for Q, K, V Weights (Layer 0)")
plt.tight_layout()

# Save to file
plt.savefig("qkv_row_col_norms.png")
