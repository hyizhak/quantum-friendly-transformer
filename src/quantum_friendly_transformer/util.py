import torch
from torch.nn import ConstantPad1d
from collections import Counter
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from transformers import TrainingArguments, Trainer, TrainerCallback, TrainerControl, TrainerState

def manual_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class LetterTokenizer:
    def __init__(self, **kwargs):
        pass

    def __call__(self, items):
        if isinstance(items, str):
            return self.__tokenize_str(items)
        else:
            return (self.__tokenize_str(t) for t in items)

    def __tokenize_str(self, t):
        tokenized = list(t.replace("\n", ""))
        tokenized.append("<eos>")
        tokenized.insert(0, "<bos>")
        tokenized.insert(0, "<cls>")
        return tokenized

def build_vocab(dataset, tokenizer, use_padding=False):
    """
    dataset: lists of (text, label) pairs
    """
    counter = Counter()

    # iterate through the dataset and count the frequency of each token
    for text, _ in dataset:
        counter.update(tokenizer(text))
    
    special_tokens = ['<cls>', '<unk>', '<bos>', '<eos>']

    if use_padding:
        special_tokens.append('<pad>')
    
    tokens = sorted(set(counter.elements()) - set(special_tokens))
    
    all_tokens = special_tokens + tokens
    
    vocab = {token: idx for idx, token in enumerate(all_tokens)}

    return vocab

def token_to_idx(token, vocab):
    return vocab.get(token, vocab['<unk>'])

def coll_factory(vocab, tokenizer, device="cpu", pad_to_length=None):
    def coll(batch):
        xs, ys = [], []

        for text, label in batch:
            ys.append(torch.tensor([label], dtype=torch.float32))
            x = torch.tensor([token_to_idx(token, vocab) for token in tokenizer(text)], dtype=torch.long)
            if pad_to_length != None:
                PAD_IDX = vocab["<pad>"]
                pad = ConstantPad1d((0, pad_to_length - len(x)), PAD_IDX)
                x = torch.tensor(pad(x), dtype=torch.long)
            xs.append(x)

        xs = torch.stack(xs)
        ys = torch.stack(ys)
        return xs.to(device), ys.to(device)

    return coll

def tokenize_and_align_labels(tokenizer, examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, padding='max_length', padding_side='right', is_split_into_words=True, max_length=64)

    labels = []
    for i, label in enumerate(examples["pos_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def tokenize_dna_sequence_gue(tokenizer, examples):
    # examples["sequence"] is a list of DNA strings
    # examples["label"] is a list of labels (or single label if not batched)
    
    # Tokenize all sequences in the batch
    tokenized_inputs = tokenizer(
        examples["sequence"],        # list of raw sequences
        padding="max_length",        # or "longest";
        truncation=True,             # truncate if sequence is too long
        max_length=96,              # max length of the tokenized sequence
    )
    
    # Add the labels into the returned dictionary
    tokenized_inputs["labels"] = examples["label"]
    
    return tokenized_inputs

def tokenize_dna_sequence_genomic_bench(tokenizer, examples):
    # examples["sequence"] is a list of DNA strings
    # examples["label"] is a list of labels (or single label if not batched)
    
    # Tokenize all sequences in the batch
    tokenized_inputs = tokenizer(
        examples["seq"],        # list of raw sequences
        padding="max_length",        # or "longest";
        truncation=True,             # truncate if sequence is too long
        max_length=72,              # max length of the tokenized sequence
    )
    
    # Add the labels into the returned dictionary
    tokenized_inputs["labels"] = examples["label"]
    
    return tokenized_inputs

def get_idx_to_label(label_list):
    return {idx: label for idx, label in enumerate(label_list)}

def get_label_to_idx(label_list):
    return {label: idx for idx, label in enumerate(label_list)}

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

    predictions = np.argmax(logits, axis=-1)

    from sklearn.metrics import f1_score, accuracy_score
    return {
        "f1": f1_score(labels, predictions),
        "accuracy": accuracy_score(labels, predictions)
    }

class PrintMetricsCallback(TrainerCallback):
    """
    A callback to automatically run after each evaluation and
    print the metrics in your custom format, e.g.:
        epoch: X, metrics: {'f1': ..., 'accuracy': ...}
    """
    def on_evaluate(
        self, 
        args: TrainingArguments, 
        state: TrainerState, 
        control: TrainerControl, 
        metrics, 
        **kwargs
    ):
        # The `metrics` dict has keys like "eval_loss", "eval_f1", "eval_accuracy", "epoch", etc.
        # We'll grab `f1` and `accuracy` if they exist.
        epoch = metrics.get("epoch", state.epoch)  # Typically stored in 'epoch'
        if epoch is None:
            epoch = 0.0
        f1 = metrics.get("eval_f1", None)
        acc = metrics.get("eval_accuracy", None)

        if f1 is not None and acc is not None:
            print(f"epoch: {int(epoch)}, metrics: {{'f1': {f1}, 'accuracy': {acc}}}")
        else:
            # If your metric keys differ or you want to see all metric keys:
            print(f"epoch: {int(epoch)}, all metrics: {metrics}")

        return control

def check_matrix_norm(model_path):
    from safetensors import safe_open

    tensors = {}
    with safe_open(model_path, framework="pt", device=0) as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)

    gamma_list = []

    for name, param in tensors.items():
        if name.endswith(".g"):
            g_value = param.data.cpu()  # move to CPU if needed
            gamma_value = torch.sigmoid(g_value)
            gamma_list.append(gamma_value)
            print(f"Parameter: {name}")
            print(f"g values:\n{g_value}")
            print(f"gamma values:\n{gamma_value}\n")

    print(np.mean(gamma_list))
    print(np.var(gamma_list))

