from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import matplotlib.pyplot as plt
import numpy as np

# Specify the model checkpoint
# model_name = "Qwen/Qwen2.5-3B-Instruct"

model_name = "meta-llama/Llama-2-7b-chat-hf"


# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

print(model)

# ------------------------------
# Part 1: Token Embeddings
# ------------------------------

# Retrieve the input embeddings.
embedding_layer = model.get_input_embeddings()
embeddings = embedding_layer.weight  # Shape: [vocab_size, embedding_dim]
print(f"Embedding matrix shape: {embeddings.shape}")

# ------------------------------
# Part 2: Vocab * Wqkv Representations
# ------------------------------

# Extract Q, K, V projection weights from the first self-attention layer.
# Note: In a linear layer, output = x @ W^T + bias.
Wq = model.model.layers[0].self_attn.q_proj.weight.detach().cpu()  # Shape: [hidden_dim, hidden_dim]
Wk = model.model.layers[0].self_attn.k_proj.weight.detach().cpu()
Wv = model.model.layers[0].self_attn.v_proj.weight.detach().cpu()

# Multiply the vocabulary embeddings by the transposed projection matrices
vocab_q = embeddings @ Wq.T
vocab_k = embeddings @ Wk.T
vocab_v = embeddings @ Wv.T

# Compute the L2 norm for each token's projected representation
vocab_q_norms = torch.norm(vocab_q, dim=1).detach()
vocab_k_norms = torch.norm(vocab_k, dim=1).detach()
vocab_v_norms = torch.norm(vocab_v, dim=1).detach()

num_bins = 30

# Helper to plot each histogram in a subplot
def plot_hist(ax, data, title):
    counts, bin_edges = np.histogram(data.cpu().numpy(), bins=num_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    ax.bar(bin_centers, counts, width=bin_edges[1] - bin_edges[0], edgecolor='black')
    ax.set_title(title, fontsize=14)
    ax.set_xlim(0, 5)
    ax.set_xlabel(r"$\ell_2$ Norm", fontsize=14)
    ax.set_ylabel("Frequency", fontsize=14)

    # Set the tick font sizes as well
    ax.tick_params(axis='both', which='major', labelsize=14)

# Create a new figure with subplots for the vocab projections
fig2, axes2 = plt.subplots(nrows=1, ncols=3, figsize=(16, 6))
plot_hist(axes2[0], vocab_q_norms, r"$\ell_2$ Norms - Q")
plot_hist(axes2[1], vocab_k_norms, r"$\ell_2$ Norms - K")
plot_hist(axes2[2], vocab_v_norms, r"$\ell_2$ Norms - V")

plt.suptitle(r"Histograms of $\ell_2$ Norms for Q/K/V Representations (Layer 0 of Llama-2-$7b$)", fontsize=14)
plt.tight_layout()
plt.savefig("vocab_qkv_norms_llama_2_7b.pdf")
plt.show()