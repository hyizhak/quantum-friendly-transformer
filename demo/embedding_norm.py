from transformers import AutoTokenizer, AutoModelForCausalLM
# from datasets import load_dataset
import torch
import matplotlib.pyplot as plt
import numpy as np
import os


# Specify the model checkpoint for Llama2
# model_name = "meta-llama/Llama-2-7b-chat-hf"

# model_name = "meta-llama/Llama-3.2-11B-Vision"

model_name = "Qwen/Qwen2.5-14B-Instruct"

# model_name = "google/gemma-3-4b-it"

# model_name = "microsoft/phi-4"

# model_name = "openai-community/gpt2-xl"

# # Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

# conll03 = load_dataset("eriktks/conll2003")

# label_list = conll03["train"].features["pos_tags"].feature.names

# model = SpectrallyNormalizedTransformerForTokenClassification(
#     d_model=4096, nhead=32, d_ff=4*4096, num_emb=tokenizer.vocab_size, max_seq_len=64, num_classes=len(label_list),
#     apply_embedding_sn=False,
#     apply_attention_sn=True,
#     apply_ffn_sn=False,
# )

# model.load_state_dict(torch.load(".../model/conll03/vanilla_epoch_20.pth"), strict=False)

print(model)

# Retrieve the input embeddings.
# embedding_layer = model.transformer.get_input_embeddings()
embedding_layer = model.get_input_embeddings()
embeddings = embedding_layer.weight  # Shape: [vocab_size, embedding_dim]
print(f"Embedding matrix shape: {embeddings.shape}")

# Compute the L2 norm of each token's embedding vector.
l2_norms = torch.norm(embeddings, dim=1).detach().cpu().numpy()
print(f"norm shape: {l2_norms.shape}")

# Compute the histogram for the L2 norms.
num_bins = 30
counts, bin_edges = np.histogram(l2_norms, bins=num_bins)

# Compute the bin centers to align the bars with their corresponding value ranges.
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# Plot the frequency bar chart (histogram)
plt.figure(figsize=(10, 6))
plt.bar(bin_centers, counts, width=bin_edges[1]-bin_edges[0], edgecolor='black')

plt.xlabel(r"$\ell_2$ Norm of Token Embeddings", fontsize=14)
plt.ylabel("Frequency (Number of Tokens)", fontsize=14)
plt.title(r"Histogram of $\ell_2$ Norms of Qwen2.5-$14b$ Token Embeddings", fontsize=14)

# Set the tick font sizes as well
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.xlim(0, 2)
plt.savefig("emb_histogram_qwen_2.5_14b_5120.pdf")
plt.show()