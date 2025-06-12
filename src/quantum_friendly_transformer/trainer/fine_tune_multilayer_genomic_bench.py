import torch
import torch.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from transformers.models.bert.configuration_bert import BertConfig
import os

from quantum_friendly_transformer.norm_transformer.util import apply_frobenius_norm
from quantum_friendly_transformer.util import tokenize_dna_sequence_genomic_bench, manual_seed, compute_metrics, PrintMetricsCallback
from quantum_friendly_transformer.norm_transformer.bert.frobenius_normalized_bert import FrobeniuslyNormalizedBertConfig, FrobeniuslyNormalizedBertForSequenceClassification


# Set the seed
manual_seed(3407)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model_name = "zhihan1996/DNABERT-2-117M"

# Load DNABert model
tokenizer = AutoTokenizer.from_pretrained(model_name)
config = BertConfig.from_pretrained(model_name)

config.attention_probs_dropout_prob = 0.1

vanilla_model = AutoModelForSequenceClassification.from_pretrained(model_name, trust_remote_code=True, config=config)

norm_config_layernorm = FrobeniuslyNormalizedBertConfig.from_bert_config(
    config,
    apply_attn_fn=True,
    apply_ffn_fn=True,
    no_layer_norm=False,
    max_gamma=2)

fn_model = FrobeniuslyNormalizedBertForSequenceClassification(norm_config_layernorm)

subnorm_config_layernorm = FrobeniuslyNormalizedBertConfig.from_bert_config(
    config,
    apply_attn_fn=True,
    apply_ffn_fn=True,
    no_layer_norm=False,
    max_gamma=1)

subnorm_model = FrobeniuslyNormalizedBertForSequenceClassification(subnorm_config_layernorm)

# Load the dataset
notata_dataset = load_dataset("katarinagresova/Genomic_Benchmarks_human_nontata_promoters")

# Preprocess the dataset
tokenized_prom = notata_dataset.map(lambda examples: tokenize_dna_sequence_genomic_bench(tokenizer, examples), batched=True).select_columns(["input_ids", "labels", "attention_mask"])

train_dataset = tokenized_prom["train"]

split = tokenized_prom["test"].train_test_split(test_size=0.5, seed=42)
val_dataset = split["train"]
test_dataset = split["test"]

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Training
for model_obj, model_label in [
    (vanilla_model, "vanilla"),
    (fn_model, "fn_model"),
    (subnorm_model, "subnorm_model")
]:
    print(f"=" * 80)
    print(model_label)

    # Define a unique output directory for each model
    current_output_dir = f".../{model_label}"
    
    # Create a new TrainingArguments instance with the unique output_dir
    current_training_args = TrainingArguments(
        output_dir=current_output_dir,
        learning_rate=8e-5,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        fp16=True,
        eval_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="steps",
        save_steps=20000,
        per_device_train_batch_size=128,
        per_device_eval_batch_size=256,
        num_train_epochs=2,
        weight_decay=0.01,
        report_to="none"
    )

    # Create a new Trainer for each model
    trainer = Trainer(
        model=model_obj,
        args=current_training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[PrintMetricsCallback()]
    )

    train_result = trainer.train()

    metrics = trainer.evaluate(test_dataset)
    print(f"{model_label} final metrics: {metrics}")

