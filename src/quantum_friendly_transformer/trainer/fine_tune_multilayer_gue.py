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
from quantum_friendly_transformer.util import tokenize_dna_sequence_gue, manual_seed, PrintMetricsCallback
from quantum_friendly_transformer.norm_transformer.bert.frobenius_normalized_bert import FrobeniuslyNormalizedBertConfig, FrobeniuslyNormalizedBertForSequenceClassification

# Set the seed
manual_seed(3407)

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

config.attention_probs_dropout_prob = 0.1

vanilla_model = AutoModelForSequenceClassification.from_pretrained(model_name, trust_remote_code=True, config=config)

norm_config = FrobeniuslyNormalizedBertConfig.from_bert_config(
    config,
    apply_attn_fn=True,
    apply_ffn_fn=True,
    no_layer_norm=True,)

# print(config)

fn_model = FrobeniuslyNormalizedBertForSequenceClassification(norm_config)

norm_config_layernorm = FrobeniuslyNormalizedBertConfig.from_bert_config(
    config,
    apply_attn_fn=True,
    apply_ffn_fn=True,
    no_layer_norm=False,
    max_gamma=2)

subnorm_config_layernorm = FrobeniuslyNormalizedBertConfig.from_bert_config(
    config,
    apply_attn_fn=True,
    apply_ffn_fn=True,
    no_layer_norm=False,
    max_gamma=1)

fn_model_layernorm = FrobeniuslyNormalizedBertForSequenceClassification(norm_config_layernorm)

subnorm_model = FrobeniuslyNormalizedBertForSequenceClassification(subnorm_config_layernorm)

# fn_model.load_state_dict(torch.load("model/modified_dna_bert_state_dict.pth"), strict=False)

# fn_model_layernorm.load_state_dict(torch.load("model/modified_dna_bert_layernorm_state_dict.pth"), strict=False)

# from safetensors import safe_open

# tensors = {}
# with safe_open("/home/users/nus/e1310988/scratch/model/multi_layer_gue/fn_layernorm/checkpoint-42400/model.safetensors", framework="pt", device=0) as f:
#     for k in f.keys():
#         tensors[k] = f.get_tensor(k)

# gamma_list = []

# for name, param in tensors.items():
#     if name.endswith(".g"):
#         g_value = param.data.cpu()  # move to CPU if needed
#         gamma_value = torch.sigmoid(g_value) * 2
#         gamma_list.append(gamma_value)
#         print(f"Parameter: {name}")
#         print(f"g values:\n{g_value}")
#         print(f"gamma values:\n{gamma_value}\n")

# print(np.mean(gamma_list))
# print(np.var(gamma_list))

subnorm_model.load_state_dict(torch.load("model/modified_dna_bert_layernorm_state_dict.pth"), strict=False)

# print(model)

# Load the dataset
notata_dataset = load_dataset("leannmlindsey/GUE", name="prom_300_notata", cache_dir=f"{cache_dir}/datasets")

# Preprocess the dataset
tokenized_prom = notata_dataset.map(lambda examples: tokenize_dna_sequence_gue(tokenizer, examples), batched=True).select_columns(["input_ids", "labels", "attention_mask"])

train_dataset = tokenized_prom["train"]
test_dataset  = tokenized_prom["test"]

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Training

def compute_metrics(eval_preds):
    """
    eval_preds is a tuple: (logits, labels).
    Something in 'logits' is causing an inhomogeneous shape, so let's print
    them carefully to see what's inside.
    """
    all_model_outputs, labels = eval_preds  # all_model_outputs is a tuple
    # all_model_outputs[0] is the real logits,
    # and all_model_outputs[1] is hidden_states
        # Check if hidden states are included. If so, logits should be the first element.
    if isinstance(all_model_outputs, (list, tuple)):
        if len(all_model_outputs) > 1:
            logits = all_model_outputs[0]
        else:
            logits = all_model_outputs
    else:
        logits = all_model_outputs

    # print("=== Debugging 'logits' structure ===")
    # print("type(logits):", type(logits))
    # if isinstance(logits, (list, tuple)):
    #     print("len(logits):", len(logits))
    #     for i, item in enumerate(logits):
    #         print(f"[{i}] type:", type(item))
    #         if hasattr(item, 'shape'):
    #             print(f"[{i}] shape:", item.shape)
    #         elif isinstance(item, (list, tuple)):
    #             print(f"[{i}] is a list/tuple of length {len(item)}")
    #         else:
    #             print(f"[{i}] can't determine shape. item:", item)
    # else:
    #     # If it's a single tensor/array, we can print shape directly
    #     if hasattr(logits, 'shape'):
    #         print("logits.shape:", logits.shape)
    
    # print("\n=== Debugging 'labels' structure ===")
    # print("type(labels):", type(labels))
    # if isinstance(labels, (list, tuple)):
    #     print("len(labels):", len(labels))
    #     for i, item in enumerate(labels):
    #         print(f"[{i}] type:", type(item))
    #         if hasattr(item, 'shape'):
    #             print(f"[{i}] shape:", item.shape)
    #         elif isinstance(item, (list, tuple)):
    #             print(f"[{i}] is a list/tuple of length {len(item)}")
    #         else:
    #             print(f"[{i}] can't determine shape. item:", item)
    # else:
    #     if hasattr(labels, 'shape'):
    #         print("labels.shape:", labels.shape)

    # # Attempt to convert them to numpy arrays AFTER printing info
    # # If it still fails, at least we'll see what's inside.
    # logits_array = np.array(logits)
    # labels_array = np.array(labels)

    # print("\n=== After converting to np.array ===")
    # print("logits_array.shape:", logits_array.shape)
    # print("labels_array.shape:", labels_array.shape)
    
    # Now the original argmax
    predictions = np.argmax(logits, axis=-1)

    from sklearn.metrics import f1_score, accuracy_score
    return {
        "f1": f1_score(labels, predictions),
        "accuracy": accuracy_score(labels, predictions)
    }

# Train each model in a loop using Trainer
for model_obj, model_label in [
    (vanilla_model, "vanilla"),
    # (fn_model, "fn_model"),
    (fn_model_layernorm, "fn_layernorm"),
    (subnorm_model, "subnorm_model")
]:
    print(f"=" * 80)
    print(model_label)

    # Define a unique output directory for each model
    current_output_dir = f"/home/users/nus/e1310988/scratch/model/multi_layer_gue/{model_label}"
    
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
        num_train_epochs=200,
        weight_decay=0.01,
        report_to="none"
    )

    # Create a new Trainer for each model
    trainer = Trainer(
        model=model_obj,
        args=current_training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[PrintMetricsCallback()]
    )

    train_result = trainer.train()

    metrics = trainer.evaluate()
    print(f"{model_label} final metrics: {metrics}")

