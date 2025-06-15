import torch
import torch.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, DataCollatorWithPadding
from transformers.models.bert.configuration_bert import BertConfig
from torch.amp import autocast, GradScaler
import os

from quantum_friendly_transformer.norm_transformer.spectral_normalized_transformer import SpectrallyNormalizedTransformerForSequenceClassification
from quantum_friendly_transformer.norm_transformer.frobenius_normalized_transformer import FrobeniuslyNormalizedTransformerForSequenceClassification
from quantum_friendly_transformer.util import tokenize_dna_sequence_gue, manual_seed
from quantum_friendly_transformer.trainer.util import train


# Set the seed
manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model_name = "zhihan1996/DNABERT-2-117M"

# Load DNABert model
tokenizer = AutoTokenizer.from_pretrained(model_name)
config = BertConfig.from_pretrained(model_name)

# one time operation: save the embedding weights
# dnabert_model = AutoModel.from_pretrained(model_name, config=config)
# embedding_weights = dnabert_model.embeddings.word_embeddings.weight.data.clone()
# torch.save(embedding_weights, "model/dnabert_embedding_weights.pth")

dnabert_embedding = nn.Embedding(num_embeddings=4096, embedding_dim=768, padding_idx=0)
with torch.no_grad():
    weights = torch.load("model/dnabert_embedding_weights.pth")
    dnabert_embedding.weight.copy_(weights)

print(dnabert_embedding)

# Load the dataset
prom_300_notata = load_dataset("leannmlindsey/GUE", name="prom_300_notata")

# Preprocess the dataset
tokenized_prom = prom_300_notata.map(lambda examples: tokenize_dna_sequence_gue(tokenizer, examples), batched=True).select_columns(["input_ids", "labels", "attention_mask"])

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

train_loader = DataLoader(tokenized_prom["train"], batch_size=32, shuffle=True, generator=torch.Generator().manual_seed(42), collate_fn=data_collator)
val_loader = DataLoader(tokenized_prom["dev"], batch_size=32, shuffle=False, collate_fn=data_collator)
test_loader = DataLoader(tokenized_prom["test"], batch_size=32, shuffle=False,  collate_fn=data_collator)

# Model
vanilla_model = SpectrallyNormalizedTransformerForSequenceClassification(
    d_model=768, nhead=12, d_ff=4*768, num_emb=tokenizer.vocab_size, num_classes=2, max_seq_len=256,
    apply_embedding_sn=False,
    apply_attention_sn=False,
    apply_ffn_sn=False,
    embedding_layer=dnabert_embedding
).to(device)

sn_model = SpectrallyNormalizedTransformerForSequenceClassification(
    d_model=768, nhead=12, d_ff=4*768, num_emb=tokenizer.vocab_size, num_classes=2, max_seq_len=256,
    apply_embedding_sn=False,
    apply_attention_sn=True,
    apply_ffn_sn=True,
    embedding_layer=dnabert_embedding
).to(device)

fn_model = FrobeniuslyNormalizedTransformerForSequenceClassification(
    d_model=768, nhead=12, d_ff=4*768, num_emb=tokenizer.vocab_size, num_classes=2, max_seq_len=256,
    apply_embedding_fn=False,
    apply_attention_fn=True,
    apply_ffn_fn=True,
    embedding_layer=dnabert_embedding
).to(device)

# Training
for model in [vanilla_model, sn_model, fn_model]:

    if model == vanilla_model:
        model_label = "vanilla"
    elif model == sn_model:
        model_label = "sn_model"
    elif model == fn_model:
        model_label = "fn_model"

    metric_fns = {'f1': f1_score, 'accuracy': accuracy_score}

    history = train(
        model,
        train_loader,
        val_loader,
        test_loader,
        device=device,
        metric_fns=metric_fns,
        save_dir=".../model/gue",
        save_prefix=model_label,
        is_sequence=False
    )