import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers.models.bert.configuration_bert import BertConfig
import os


model_name = "zhihan1996/DNABERT-2-117M"

tokenizer = AutoTokenizer.from_pretrained(model_name)
config = BertConfig.from_pretrained(model_name)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name, 
    trust_remote_code=True, 
    config=config
)

model_state_dict = model.state_dict()
print("Original model state dict keys:\n", list(model_state_dict.keys()))

# Create a new state dict for the modified model
new_state_dict = {}

for key, value in model_state_dict.items():
    # 1) Skip any layernorm keys (covers "LayerNorm" and "layernorm")
    # if "layernorm" in key.lower():
    #     continue

    # 2) Chunk Wqkv into Wq, Wk, Wv and rename them to "Wq.parametrizations.weight.original", etc.
    if "Wqkv" in key:
        if "weight" in key:
            # Split (3 * hidden_size, hidden_size) into three (hidden_size, hidden_size).
            q_weight, k_weight, v_weight = value.chunk(3, dim=0)

            # Construct the base by removing "Wqkv.weight"
            # e.g. "bert.encoder.layer.0.attention.self.Wqkv.weight" -> "bert.encoder.layer.0.attention.self."
            base = key.replace(".Wqkv.weight", ".")
            # New names with "parametrizations.weight.original"
            new_state_dict[base + "Wq.parametrizations.weight.original"] = q_weight
            new_state_dict[base + "Wk.parametrizations.weight.original"] = k_weight
            new_state_dict[base + "Wv.parametrizations.weight.original"] = v_weight

        elif "bias" in key:
            # Split (3 * hidden_size,) into three (hidden_size,)
            q_bias, k_bias, v_bias = value.chunk(3, dim=0)

            base = key.replace(".Wqkv.bias", ".")
            # The new model typically does NOT expect biases to have "parametrizations"
            new_state_dict[base + "Wq.bias"] = q_bias
            new_state_dict[base + "Wk.bias"] = k_bias
            new_state_dict[base + "Wv.bias"] = v_bias

    else:
        # 3) For layers that the new model expects in param form:
        #    - Typically that means attention.output.dense.weight, mlp.gated_layers.weight, mlp.wo.weight, 
        #      plus the new Wq/Wk/Wv we just created.
        #
        #    However, the new error suggests embeddings, pooler, classifier 
        #    are STILL expected in the standard (non-param) naming.

        # We can do a small helper function that returns True for keys we want to keep in param form:
        def should_use_parametrizations(k: str) -> bool:
            # These are the typical BERT layers we rename to param form:
            #   "attention.output.dense.weight",
            #   "mlp.gated_layers.weight",
            #   "mlp.wo.weight".
            # But we DO NOT do so for embeddings, pooler, classifier
            if any(x in k for x in ["embeddings.word_embeddings", 
                                    "embeddings.token_type_embeddings",
                                    "pooler.dense",
                                    "classifier",
                                    "layernorm",
                                    "LayerNorm"]):
                return False
            # Also do not rename biases to param form
            if k.endswith(".bias"):
                return False
            # If it ends with .weight, we want param form
            if k.endswith(".weight"):
                return True
            return False

        if should_use_parametrizations(key):
            # rename e.g. ".weight" -> ".parametrizations.weight.original"
            new_key = key.replace(".weight", ".parametrizations.weight.original")
            new_state_dict[new_key] = value
        else:
            # keep the original name as is
            new_state_dict[key] = value

torch.save(new_state_dict, "model/modified_dna_bert_layernorm_state_dict.pth")
print("\nModified state dict keys:\n", list(new_state_dict.keys()))