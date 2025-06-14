import sys

import numpy as np
import torch
from datasets import load_dataset
from numpy.linalg import norm
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import dask.array as da


model_card_embedding = {
    "bert": "bert-base-uncased",
    "distilgpt": "distilbert/distilgpt2",
    "bart": "facebook/bart-base",
    "roberta": "FacebookAI/roberta-base",
    "gpt2": "openai-community/gpt2",
    "gpt": "openai-community/openai-gpt",
    "llama2-7b": "meta-llama/Llama-2-7b-chat-hf",
    "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "mistral7b": "mistralai/Mistral-7B-v0.1",
}

torch.cuda.empty_cache()


def mmlu():
    dataset = load_dataset("cais/mmlu", "all")
    mmlu_validation = []

    for data in dataset["validation"]:
        mmlu_validation.append(data["question"])

    return mmlu_validation


def random_dataset(max_tokens, tokenizer_size, tokenizer_special_tokens):
    dataset = []

    all_tokens = np.arange(0, tokenizer_size)
    valid_tokens = [x for x in all_tokens if x not in tokenizer_special_tokens]
    
    for i in range(1, max_tokens + 1):
        random_input_ids = torch.from_numpy(np.random.choice(valid_tokens, size=(1, i)))
        attention_mask = torch.ones((1, i))
        inputs = {
            "input_ids": random_input_ids,
            "attention_mask": attention_mask,
        }
        dataset.append(inputs)

    return dataset


def batch_inputs(dataset, batch_size):
    batch_mat = []
    batches = len(dataset) // batch_size
    for i in range(batches + 1):
        if i == batches:
            batch_mat.append(dataset[i * batch_size :])
        else:
            batch_mat.append(dataset[i * batch_size : (i + 1) * batch_size])

    return batch_mat


def get_token_length_arr(batched_inputs, tokenizer):
    token_length_arr = []

    for batch in batched_inputs:
        batch_arr = []
        for data in batch:
            input = tokenizer(data)
            batch_arr.append(len(input["input_ids"]))
        token_length_arr.append(batch_arr)

    return token_length_arr


def main(name, max_length, batch_size, random=False):

    model_name = model_card_embedding[name]

    if random:
        dataset_str = "on random dataset"
    else:
        dataset_str = "on mmlu dataset"

    print(
        f"starting inference for {model_name} with batch_size {batch_size} "
        + dataset_str
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if name in ["llama2-7b", "mistral7b", "tinyllama"]:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="balanced_low_0"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    model.resize_token_embeddings(len(tokenizer))

    if random:
        dataset = random_dataset(512, len(tokenizer), tokenizer.all_special_ids)
        token_length_arr = np.arange(1, 513).reshape(512, 1)
    else:
        dataset = mmlu()
        dataset = batch_inputs(dataset, batch_size)
        token_length_arr = get_token_length_arr(dataset, tokenizer)

    embed_norms_arr = []
    first_layer_norms_arr = []
    embed_spec_arr = []


    for batch in tqdm(dataset):
        if random:
            inputs = batch
            batch_size = 1
        elif max_length == -1:
            inputs = tokenizer.batch_encode_plus(
                batch, padding=True, return_tensors="pt"
            )
        else:
            inputs = tokenizer.batch_encode_plus(
                batch,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )

        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states

        embeddings = hidden_states[0].detach().cpu()
        first_layer = hidden_states[1].detach().cpu()

        # frobenius norm
        embed_norm = norm(embeddings, ord="fro", axis=(1, 2))
        first_layer_norm = norm(first_layer, ord="fro", axis=(1, 2))

        # spectral norm
        if name in ["llama2-7b", "mistral7b", "tinyllama"]:
            embed_spec = []
            for i in range(batch_size):
                dask_array = da.from_array(embeddings[0, :, :].numpy(), chunks=120)
                u, s, v = da.linalg.svd_compressed(dask_array, k=5)
                embed_spec.append(s.compute()[0])
        else:
            embed_spec = np.linalg.svd(embeddings)[1][:, 0]

        embed_norms_arr.append(embed_norm)
        first_layer_norms_arr.append(first_layer_norm)
        embed_spec_arr.append(embed_spec)

    data_arr = []

    for i in range(len(embed_norms_arr)):
        for j in range(len(embed_norms_arr[i])):
            data_arr.append(
                (
                    token_length_arr[i][j],
                    embed_norms_arr[i][j],
                    first_layer_norms_arr[i][j],
                    embed_spec_arr[i][j],
                )
            )
    
    data_arr = np.asarray(data_arr)

    if random:
        np.save(f"data_files/{name}_random_data.npy", data_arr)
    else:
        np.save(f"data_files/{name}_data.npy", data_arr)


if __name__ == "__main__":
    model_name = sys.argv[1]
    max_length = int(sys.argv[2])
    batch_size = int(sys.argv[3])
    random = bool(int(sys.argv[4]))
    main(model_name, max_length, batch_size, random)
