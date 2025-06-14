from transformers import AutoModelForCausalLM, AutoTokenizer

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

if __name__ == '__main__':
    for k, v in model_card_embedding.items():
        print(k)
        model = AutoModelForCausalLM.from_pretrained(v)
        t = AutoTokenizer.from_pretrained(v)
    