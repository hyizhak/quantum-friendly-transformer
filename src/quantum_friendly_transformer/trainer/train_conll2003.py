from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForTokenClassification
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import os
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

from quantum_friendly_transformer.norm_transformer.spectral_normalized_transformer import SpectrallyNormalizedTransformerForTokenClassification
from quantum_friendly_transformer.norm_transformer.frobenius_normalized_transformer import FrobeniuslyNormalizedTransformerForTokenClassification
from quantum_friendly_transformer.util import tokenize_and_align_labels, manual_seed
from quantum_friendly_transformer.trainer.util import train


# Set the seed
manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

conll03 = load_dataset("eriktks/conll2003")

label_list = conll03["train"].features["pos_tags"].feature.names

# Specify the model checkpoint for Llama2
model_name = "meta-llama/Llama-2-7b-chat-hf"

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

llama_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)

# Retrieve the input embeddings.
embedding_layer = llama_model.get_input_embeddings()

# Preprocess the dataset
tokenized_conll03 = conll03.map(lambda examples: tokenize_and_align_labels(tokenizer, examples), batched=True).select_columns(["input_ids", "labels", "attention_mask"])

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

train_loader = DataLoader(tokenized_conll03["train"], batch_size=32, shuffle=True, generator=torch.Generator().manual_seed(42), collate_fn=data_collator)
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

fn_model = FrobeniuslyNormalizedTransformerForTokenClassification(
    d_model=4096, nhead=32, d_ff=4*4096, num_emb=tokenizer.vocab_size, max_seq_len=64, num_classes=len(label_list),
    apply_embedding_fn=False,
    apply_attention_fn=True,
    apply_ffn_fn=True,
    embedding_layer=embedding_layer
).to(device)

# Training
for model in [vanilla_model, sn_model, fn_model]:

    if model == vanilla_model:
        model_label = "vanilla"
    elif model == sn_model:
        model_label = "sn_model"
    elif model == fn_model:
        model_label = "fn_model"

    metric_fns = {'f1': lambda y_true, y_pred: f1_score(y_true, y_pred, average='macro'), 'accuracy': accuracy_score}

    history = train(
        model,
        train_loader,
        val_loader,
        test_loader,
        device=device,
        metric_fns=metric_fns,
        save_dir=".../model/conll03",
        save_prefix=model_label,
    )