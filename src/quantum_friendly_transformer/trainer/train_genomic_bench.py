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
from quantum_friendly_transformer.util import tokenize_dna_sequence_genomic_bench, manual_seed

# Set the seed
manual_seed(42)

cache_dir = "/home/users/nus/e1310988/scratch/huggingface"

os.environ['HF_HOME'] = cache_dir
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# model_name = "zhihan1996/DNABERT-2-117M"
model_name = f"{cache_dir}/hub/DNABERT-2-117M"

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

# Load the dataset
notata_dataset = load_dataset("katarinagresova/Genomic_Benchmarks_human_nontata_promoters", cache_dir=f"{cache_dir}/datasets")

# # Preprocess the dataset
tokenized_prom = notata_dataset.map(lambda examples: tokenize_dna_sequence_genomic_bench(tokenizer, examples), batched=True).select_columns(["input_ids", "labels", "attention_mask"])

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

train_loader = DataLoader(tokenized_prom["train"], batch_size=32, shuffle=True, collate_fn=data_collator)

split = tokenized_prom["test"].train_test_split(test_size=0.5, seed=42)
val_dataset = split["train"]
test_dataset = split["test"]

val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=data_collator)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False,  collate_fn=data_collator)

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
        model_name = "vanilla"
    elif model == sn_model:
        model_name = "sn_model"
    elif model == fn_model:
        model_name = "fn_model"

    print("=" * 80)
    print(model_name)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-6)
    scaler = GradScaler()

    acc = []

    for epoch in tqdm(range(1, 21)):
        model.train()
        for i, batch in enumerate(train_loader):
            with autocast(device_type=str(device)):
                batch = {k: v.to(device) for k, v in batch.items()}
                x, y, attn_mask = batch["input_ids"], batch["labels"], batch["attention_mask"]

                optimizer.zero_grad()
                y_pred = model(x, key_padding_mask=(attn_mask == 0))
                y = y.squeeze()
                loss = criterion(y_pred, y.long())
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        # Evaluation
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                x, y, attn_mask = batch["input_ids"], batch["labels"], batch["attention_mask"]
                y_pred = model(x, key_padding_mask=(attn_mask == 0))
                
                all_preds.extend(torch.argmax(y_pred, dim=1).tolist())
                all_labels.extend(y.tolist())

        metrics = {
            'f1': f1_score(all_labels, all_preds),
            'accuracy': accuracy_score(all_labels, all_preds)
        }

        print(f"epoch: {epoch}, metrics: {metrics}")

        if epoch <= 30:
            if epoch % 10 == 0:
                torch.save(model.state_dict(), f"/home/users/nus/e1310988/scratch/model/genomic_bench/{model_name}_epoch_{epoch}.pth")
        elif epoch % 40 == 0:
            torch.save(model.state_dict(), f"/home/users/nus/e1310988/scratch/model/genomic_bench/{model_name}_epoch_{epoch}.pth")

    # Final Evaluation
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            x, y, attn_mask = batch["input_ids"], batch["labels"], batch["attention_mask"]
            y_pred = model(x, key_padding_mask=(attn_mask == 0))
            
            all_preds.extend(torch.argmax(y_pred, dim=1).tolist())
            all_labels.extend(y.tolist())
    metrics = {
        'f1': f1_score(all_labels, all_preds),
        'accuracy': accuracy_score(all_labels, all_preds)
    }
    print(f'{model_name} final metrics: {metrics}')