from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForTokenClassification
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..spectral_norm_transformer.spectral_normalized_transformer_block import SpectrallyNormalizedTransformerForTokenClassification
from ..util import tokenize_and_align_labels, compute_metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

conll03 = load_dataset("eriktks/conll2003")

label_list = conll03["train"].features["pos_tags"].feature.names

# Specify the model checkpoint for Llama2
model_name = "meta-llama/Llama-2-7b-chat-hf"

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

llama_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

# Retrieve the input embeddings.
embedding_layer = llama_model.get_input_embeddings()

# Preprocess the dataset
tokenized_conll03 = conll03.map(lambda examples: tokenize_and_align_labels(tokenizer, examples), batched=True).select_columns(["input_ids", "labels"])

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer, padding='max_length', max_length=64)

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

    model_name = model.__class__.__name__

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    for epoch in tqdm(range(20)):
        model.train()
        for i, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            x, y = batch["input_ids"], batch["labels"]

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            optimizer.step()

        torch.save(model.state_dict(), f"./model/{model_name}_conll03_epoch_{epoch}.pth")

        # Evaluation
        model.eval()
        label_preds = []
        label_true = []
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            x, y = batch["input_ids"], batch["labels"]
            logits = model(x)
            label_preds.extend(torch.argmax(logits, dim=-1).view(-1).tolist())
            label_true.extend(y.view(-1).tolist())

        metrics = compute_metrics(label_list, (label_preds, label_true))
        print(f"Epoch {epoch}")
        print(metrics)
