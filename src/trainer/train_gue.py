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

from spectral_norm_transformer.spectral_normalized_transformer_block import SpectrallyNormalizedTransformerForSequenceClassification
from src.util import tokenize_dna_sequence, manual_seed

# Set the seed
manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load DNABert model
tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M")
config = BertConfig.from_pretrained("zhihan1996/DNABERT-2-117M")
dnabert_model = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True, config=config)

# one time operation: save the embedding weights
embedding_weights = dnabert_model.embeddings.word_embeddings.weight.data.clone()

torch.save(embedding_weights, "model/dnabert_embedding_weights.pt")

dnabert_embedding = nn.Embedding(num_embeddings=4096, embedding_dim=768, padding_idx=0)
with torch.no_grad():
    weights = torch.load("model/dnabert_embedding_weights.pt")
    dnabert_embedding.weight.copy_(weights)

# Load the dataset
prom_300_notata = load_dataset("leannmlindsey/GUE", name="prom_300_notata")

# # Preprocess the dataset
tokenized_prom = prom_300_notata.map(lambda examples: tokenize_dna_sequence(tokenizer, examples), batched=True).select_columns(["input_ids", "labels", "attention_mask"])

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

train_loader = DataLoader(tokenized_prom["train"], batch_size=32, shuffle=True, collate_fn=data_collator)
val_loader = DataLoader(tokenized_prom["dev"], batch_size=32, shuffle=False, collate_fn=data_collator)
test_loader = DataLoader(tokenized_prom["test"], batch_size=32, shuffle=False,  collate_fn=data_collator)

# # Model
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

# # Training
for model in [vanilla_model, sn_model]:

    model_name = "vanilla" if model == vanilla_model else "sn_model"

    print("=" * 80)
    print(model_name)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    scaler = GradScaler()

    acc = []

    for epoch in tqdm(range(400)):
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

        print(f"epoch: {epoch+1}, metrics: {metrics}")

        if epoch <= 30:
            if epoch % 10 == 0:
                torch.save(model.state_dict(), f"/home/users/nus/e1310988/scratch/model/gue/{model_name}_epoch_{epoch}.pth")
        elif epoch % 40 == 0:
            torch.save(model.state_dict(), f"/home/users/nus/e1310988/scratch/model/gue/{model_name}_epoch_{epoch}.pth")