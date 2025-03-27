from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import torch
import matplotlib.pyplot as plt
import numpy as np
import os

from norm_transformer.spectral_normalized_transformer_block import SpectrallyNormalizedTransformerForTokenClassification

# cache_dir = "/home/users/nus/e1310988/scratch/huggingface"

# os.environ['HF_HOME'] = cache_dir
# os.environ['HF_DATASETS_OFFLINE'] = '1'
# os.environ['HF_HUB_OFFLINE'] = '1'

# Specify the model checkpoint for Llama2
# model_name = "meta-llama/Llama-2-7b-chat-hf"
# model_name = f"{cache_dir}/hub/Llama-2-7b-chat-hf"

# model_name = "meta-llama/Llama-3.2-11B-Vision"

# model_name = "Qwen/Qwen2.5-14B-Instruct"

model_name = "google/gemma-2-2b-it"

# # Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

# conll03 = load_dataset("eriktks/conll2003", cache_dir=f"{cache_dir}/datasets")

# label_list = conll03["train"].features["pos_tags"].feature.names

# model = SpectrallyNormalizedTransformerForTokenClassification(
#     d_model=4096, nhead=32, d_ff=4*4096, num_emb=tokenizer.vocab_size, max_seq_len=64, num_classes=len(label_list),
#     apply_embedding_sn=False,
#     apply_attention_sn=True,
#     apply_ffn_sn=False,
# )

# model.load_state_dict(torch.load("/home/users/nus/e1310988/scratch/model/conll03/vanilla_epoch_20.pth"), strict=False)

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
plt.xlabel("L2 Norm of Token Embeddings")
plt.ylabel("Frequency (Number of Tokens)")
plt.xlim(0, 4)
plt.title("Histogram of L2 Norms of gemma-2-2b Token Embeddings")
plt.savefig("gemma_2b_2304_emb_histogram.png")