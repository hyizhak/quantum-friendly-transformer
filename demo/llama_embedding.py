from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import matplotlib.pyplot as plt
import numpy as np

# Specify the model checkpoint for Llama2
model_name = "meta-llama/Llama-2-7b-chat-hf"

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

# Retrieve the input embeddings.
embedding_layer = model.get_input_embeddings()
embeddings = embedding_layer.weight  # Shape: [vocab_size, embedding_dim]
print(f"Embedding matrix shape: {embeddings.shape}")

# Compute the L2 norm of each token's embedding vector.
l2_norms = torch.norm(embeddings, dim=1).detach().cpu().numpy()
print(f"norm shape: {l2_norms.shape}")

# Get the vocabulary mapping token -> id
vocab = tokenizer.get_vocab()

# Create a list of tokens sorted by their token id (so that indices match the embedding matrix rows)
tokens = sorted(vocab, key=lambda token: vocab[token])

# Verify that the number of tokens matches the number of embedding rows
assert len(tokens) == embeddings.size(0), "Mismatch between vocab size and embedding matrix rows."

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
# plt.title("Histogram of L2 Norms of Llama2 Token Embeddings")
plt.savefig("llama_emb_histogram.png")