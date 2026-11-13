import argparse
import json
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from numpy.linalg import norm
from tqdm import tqdm
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

if __package__:
    from .norm_data import validate_embedding_norms
else:
    from norm_data import validate_embedding_norms


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

def mmlu():
    dataset = load_dataset("cais/mmlu", "all")
    mmlu_validation = []

    for data in dataset["validation"]:
        mmlu_validation.append(data["question"])

    return mmlu_validation


def random_dataset(max_tokens, tokenizer_size, tokenizer_special_tokens, *, seed=0):
    """One independently drawn sequence for each length, with a local RNG."""
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    dataset = []
    rng = np.random.default_rng(seed)
    all_tokens = np.arange(0, tokenizer_size)
    valid_tokens = np.setdiff1d(all_tokens, tokenizer_special_tokens)
    if not len(valid_tokens):
        raise ValueError("Tokenizer has no non-special tokens")
    
    for i in range(1, max_tokens + 1):
        random_input_ids = torch.from_numpy(rng.choice(valid_tokens, size=(1, i)))
        attention_mask = torch.ones((1, i), dtype=torch.long)
        inputs = {
            "input_ids": random_input_ids,
            "attention_mask": attention_mask,
        }
        dataset.append(inputs)

    return dataset


def batch_inputs(dataset, batch_size):
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return [dataset[i : i + batch_size] for i in range(0, len(dataset), batch_size)]


def get_token_length_arr(batched_inputs, tokenizer):
    token_length_arr = []

    for batch in batched_inputs:
        batch_arr = []
        for data in batch:
            input = tokenizer(data)
            batch_arr.append(len(input["input_ids"]))
        token_length_arr.append(batch_arr)

    return token_length_arr


def measure_batch_norms(embeddings, first_layer, attention_mask):
    """Measure each sample's valid tokens using an exact spectral norm."""
    if embeddings.shape != first_layer.shape or attention_mask.shape != embeddings.shape[:2]:
        raise ValueError("Hidden states and attention mask must describe the same batch")
    embeddings = embeddings.detach().cpu().double().numpy()
    first_layer = first_layer.detach().cpu().double().numpy()
    masks = attention_mask.detach().cpu().bool().numpy()
    rows = []
    for embedding, hidden, mask in zip(embeddings, first_layer, masks):
        if not mask.any():
            raise ValueError("Cannot measure a sample without valid tokens")
        embedding, hidden = embedding[mask], hidden[mask]
        rows.append((int(mask.sum()), norm(embedding, ord="fro"),
                     norm(hidden, ord="fro"), np.linalg.svd(embedding, compute_uv=False)[0]))
    return validate_embedding_norms(np.asarray(rows))


def main(name, max_length, batch_size, random=False, *, model_path=None,
         output_dir=None, device=None, first_two_layers=False, questions=None, seed=0):
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if max_length != -1 and max_length <= 0:
        raise ValueError("max_length must be positive or -1")
    if first_two_layers and name not in ["llama2-7b", "mistral7b", "tinyllama"]:
        raise ValueError("first_two_layers is supported only for Llama and Mistral models")
    model_name = model_path or model_card_embedding[name]

    if random:
        dataset_str = "on random dataset"
    else:
        dataset_str = "on mmlu dataset"

    print(
        f"starting inference for {model_name} with batch_size {batch_size} "
        + dataset_str
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_options = {"torch_dtype": torch.float32}
    if device is not None:
        model_options["device_map"] = device
    elif name in ["llama2-7b", "mistral7b", "tinyllama"] and torch.cuda.is_available():
        model_options["device_map"] = "balanced_low_0"
    if first_two_layers:
        # With two decoder layers, hidden_states[1] remains the output of the
        # first decoder layer, before the model's final normalization.
        model_options["num_hidden_layers"] = 2
    # Loading bart-base as BartForCausalLM leaves its shared embedding table
    # uninitialized in some Transformers versions. Measure its encoder using
    # the checkpoint's native architecture, including the shared embeddings.
    model_class = AutoModel if name == "bart" else AutoModelForCausalLM
    model = model_class.from_pretrained(model_name, **model_options)
    model.eval()
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    model.resize_token_embeddings(len(tokenizer))
    if name == "bart":
        model = model.get_encoder()

    if random:
        dataset = random_dataset(512 if max_length == -1 else max_length,
                                 len(tokenizer), tokenizer.all_special_ids, seed=seed)
    else:
        dataset = mmlu() if questions is None else questions
        dataset = batch_inputs(dataset, batch_size)
    measured_batches = []
    for batch in tqdm(dataset):
        if random:
            inputs = batch
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

        inputs = {key: value.to(model.device) for key, value in inputs.items()}
        with torch.inference_mode():
            forward_options = {} if name == "bart" else {"use_cache": False}
            outputs = model(**inputs, output_hidden_states=True, **forward_options)
        measured_batches.append(measure_batch_norms(
            outputs.hidden_states[0], outputs.hidden_states[1], inputs["attention_mask"],
        ))

    if not measured_batches:
        raise ValueError("Cannot measure an empty dataset")
    data_arr = validate_embedding_norms(np.concatenate(measured_batches), source=f"{name} measurements")
    output_dir = Path(output_dir) if output_dir is not None else Path(__file__).parent / "data_files"
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_random" if random else ""
    np.save(output_dir / f"{name}{suffix}_data.npy", data_arr)
    if random:
        # Persist the actual draws as well as the seed: RNG implementations and
        # tokenizer vocabularies may change between software/model revisions.
        (output_dir / f"{name}_random_inputs.json").write_text(json.dumps({
            "seed": seed, "rng": "numpy.default_rng/PCG64",
            "input_ids": [sample["input_ids"][0].tolist() for sample in dataset],
        }, separators=(",", ":")) + "\n")
    return data_arr


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure per-sample input matrix norms")
    parser.add_argument("model_name", choices=model_card_embedding)
    parser.add_argument("max_length", type=int)
    parser.add_argument("batch_size", type=int)
    parser.add_argument("random", type=int, choices=[0, 1])
    parser.add_argument("--model-path", help="Use an existing local model snapshot")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", help="For example cpu or cuda:0")
    parser.add_argument("--first-two-layers", action="store_true")
    parser.add_argument("--questions-json", type=Path, help="JSON list of questions for offline MMLU runs")
    parser.add_argument("--seed", type=int, default=0, help="Seed for random token sequences")
    args = parser.parse_args()
    questions = json.loads(args.questions_json.read_text()) if args.questions_json else None
    main(args.model_name, args.max_length, args.batch_size, bool(args.random),
         model_path=args.model_path, output_dir=args.output_dir, device=args.device,
         first_two_layers=args.first_two_layers, questions=questions, seed=args.seed)
