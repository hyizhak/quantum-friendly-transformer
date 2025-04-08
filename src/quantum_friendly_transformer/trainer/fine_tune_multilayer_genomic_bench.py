import torch
import torch.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from datasets import load_dataset
from transformers import AutoTokenizer, TrainingArguments, Trainer, DataCollatorWithPadding
from transformers.models.bert.configuration_bert import BertConfig
from torch.amp import autocast, GradScaler
import os

from quantum_friendly_transformer.norm_transformer.util import apply_frobenius_norm
from quantum_friendly_transformer.util import tokenize_dna_sequence_genomic_bench, manual_seed
from quantum_friendly_transformer.norm_transformer.bert.frobenius_normalized_bert import NormalizedBertConfig, BertForSequenceClassification

# Set the seed
manual_seed(42)

# cache_dir = "/home/users/nus/e1310988/scratch/huggingface"

# os.environ['HF_HOME'] = cache_dir
# os.environ['HF_DATASETS_OFFLINE'] = '1'
# os.environ['HF_HUB_OFFLINE'] = '1'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model_name = "zhihan1996/DNABERT-2-117M"
# model_name = f"{cache_dir}/hub/DNABERT-2-117M"

# Load DNABert model
tokenizer = AutoTokenizer.from_pretrained(model_name)
config = BertConfig.from_pretrained(model_name)
norm_config = NormalizedBertConfig.from_bert_config(
    config,
    apply_attn_fn=True,
    apply_ffn_fn=True,
    no_layer_norm=True,)

print(config)

model = BertForSequenceClassification(norm_config)

model.load_state_dict(torch.load("model/modified_dna_bert_state_dict.pth"))

print(model)

# Load the dataset
notata_dataset = load_dataset("katarinagresova/Genomic_Benchmarks_human_nontata_promoters")

# # Preprocess the dataset
tokenized_prom = notata_dataset.map(lambda examples: tokenize_dna_sequence_genomic_bench(tokenizer, examples), batched=True).select_columns(["input_ids", "labels", "attention_mask"])

train_dataset = tokenized_prom["train"]
test_dataset  = tokenized_prom["test"]

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Training

def compute_metrics(eval_preds):
    """
    eval_preds is a tuple: (logits, labels)
    We want to compute F1 and accuracy.
    """
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return {
        "f1": f1_score(labels, predictions),
        "accuracy": accuracy_score(labels, predictions)
    }

training_args = TrainingArguments(
    output_dir="outputs",
    learning_rate=2e-6,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    fp16=True,
    evaluation_strategy="epoch",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    num_train_epochs=50,
    weight_decay=0.01,
    report_to="none"
)

# 12) Train each model in a loop using Trainer
for model_obj, model_label in [
    (model, "fn_model")
]:
    print(f"\n\n==========================================")
    print(f"Now training: {model_label}")

    # Create a new Trainer for each model
    trainer = Trainer(
        model=model_obj,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    train_result = trainer.train()

    metrics = trainer.evaluate()
    print(f"{model_label} final metrics: {metrics}")

