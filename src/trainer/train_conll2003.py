from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForTokenClassification
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import os
import numpy as np

from spectral_norm_transformer.spectral_normalized_transformer_block import SpectrallyNormalizedTransformerForTokenClassification
from src.util import tokenize_and_align_labels, compute_metrics, manual_seed

# Set the seed
manual_seed(42)

cache_dir = "/home/users/nus/e1310988/scratch/huggingface"

os.environ['HF_HOME'] = cache_dir
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

conll03 = load_dataset("eriktks/conll2003", cache_dir=f"{cache_dir}/datasets")

label_list = conll03["train"].features["pos_tags"].feature.names

# Specify the model checkpoint for Llama2
# model_name = "meta-llama/Llama-2-7b-chat-hf"
model_name = f"{cache_dir}/hub/Llama-2-7b-chat-hf"

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

llama_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)

# Retrieve the input embeddings.
embedding_layer = llama_model.get_input_embeddings()

# Preprocess the dataset
tokenized_conll03 = conll03.map(lambda examples: tokenize_and_align_labels(tokenizer, examples), batched=True).select_columns(["input_ids", "labels", "attention_mask"])

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

train_loader = DataLoader(tokenized_conll03["train"], batch_size=32, shuffle=True, collate_fn=data_collator)
val_loader = DataLoader(tokenized_conll03["validation"], batch_size=32, shuffle=False, collate_fn=data_collator)
test_loader = DataLoader(tokenized_conll03["test"], batch_size=32, shuffle=False,  collate_fn=data_collator)

# Model
vanilla_model = SpectrallyNormalizedTransformerForTokenClassification(
    d_model=4096, nhead=32, d_ff=4*4096, num_emb=tokenizer.vocab_size, max_seq_len=64, num_classes=len(label_list),
    apply_embedding_sn=False,
    apply_attention_sn=False,
    apply_ffn_sn=False,
    embedding_layer=embedding_layer
).to(device)

sn_model = SpectrallyNormalizedTransformerForTokenClassification(
    d_model=4096, nhead=32, d_ff=4*4096, num_emb=tokenizer.vocab_size, max_seq_len=64, num_classes=len(label_list),
    apply_embedding_sn=False,
    apply_attention_sn=True,
    apply_ffn_sn=True,
    embedding_layer=embedding_layer
).to(device)

# Training
for model in [vanilla_model, sn_model]:

    model_name = "vanilla" if model == vanilla_model else "sn_model"

    print("=" * 80)
    print(model_name)

    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
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
                logits = model(x, key_padding_mask=(attn_mask == 0))
                loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
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
                logits = model(x, key_padding_mask=(attn_mask == 0))
                # Argmax over the last dimension (num_labels) => shape [batch_size, seq_length]
                preds = torch.argmax(logits, dim=-1)
                
                # Move to CPU for metric computation
                preds = preds.detach().cpu().numpy()
                labels = y.detach().cpu().numpy()

                # Append to list (each element is shape [batch_size, seq_length])
                all_preds.append(preds)
                all_labels.append(labels)

        # Concatenate along batch dimension => final shape [total_samples, seq_length]
        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        # Now pass (preds, labels) to compute_metrics
        metrics = compute_metrics(label_list, (all_preds, all_labels))
        print(f'epoch: {epoch}, metrics: {metrics}')
        acc.append(metrics['accuracy'])

        if epoch <= 30:
            if epoch % 10 == 0:
                torch.save(model.state_dict(), f"/home/users/nus/e1310988/scratch/model/conll03/{model_name}_epoch_{epoch}.pth")
        elif epoch % 40 == 0:
            torch.save(model.state_dict(), f"/home/users/nus/e1310988/scratch/model/conll03/{model_name}_epoch_{epoch}.pth")
