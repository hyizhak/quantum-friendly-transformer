import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm
from torch.nn import ConstantPad1d
from collections import Counter


def apply_spectral_norm(module, layers=(nn.Linear,)):
    """
    Recursively apply spectral normalization to the specified layer types.
    
    Args:
        module (nn.Module): The model or submodule to modify.
        layers (tuple): Layer classes to apply SN to.
    """

    def SN(module):
        if isinstance(module, layers):
            module = spectral_norm(module)

    module.apply(SN)

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




