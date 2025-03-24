import torch
from torch.nn import ConstantPad1d
from collections import Counter
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

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

def tokenize_dna_sequence(tokenizer, examples):
    # examples["sequence"] is a list of DNA strings
    # examples["label"] is a list of labels (or single label if not batched)
    
    # Tokenize all sequences in the batch
    tokenized_inputs = tokenizer(
        examples["sequence"],        # list of raw sequences
        padding="max_length",        # or "longest";
        truncation=True,             # truncate if sequence is too long
        max_length=256,              # max length of the tokenized sequence
    )
    
    # Add the labels into the returned dictionary
    tokenized_inputs["labels"] = examples["label"]
    
    return tokenized_inputs

def get_idx_to_label(label_list):
    return {idx: label for idx, label in enumerate(label_list)}

def get_label_to_idx(label_list):
    return {label: idx for idx, label in enumerate(label_list)}

def compute_metrics(label_list, p):
    predictions, labels = p

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    # Now flatten them into 1D lists of token-level predictions & labels
    flat_preds = [token for seq in true_predictions for token in seq]
    flat_labels = [token for seq in true_labels for token in seq]

    # Finally, call classification_report on the flattened lists
    report = {
        'f1': f1_score(flat_labels, flat_preds, average='macro'),
        'accuracy': accuracy_score(flat_labels, flat_preds)
    }
    return report




