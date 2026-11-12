"""Apply the recorded recovery protocol to OLMo and Mistral.

OLMo trains all float32 master parameters. Mistral retains the archived
head/norm/cap training scope with bfloat16 frozen backbone matrices. Both use
gradient checkpointing and the same fixed, bounded attention denominator.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

import torch
from torch import nn
from transformers import AutoModel
from transformers.modeling_outputs import SequenceClassifierOutput

from experiments.lib.recovery_extension import AttentionInputMonitor
from experiments.runners import reviewer_recovery_extension as core
from quantum_friendly_transformer.experiments.runners.recovery.glue_sequence_classifier import (
    configure_trainable, make_decoder_sequence_classifier,
)
from quantum_friendly_transformer.norm_transformer import apply_attention_scaling, apply_learnable_frobenius_caps


class LargeAttentionInputMonitor(AttentionInputMonitor):
    def __enter__(self):
        for name, module in self.model.named_modules():
            if type(module).__name__ in {'OlmoAttention', 'Olmo2Attention', 'MistralAttention'}:
                self.handles.append(module.register_forward_pre_hook(self._hook(name), with_kwargs=True))
        if not self.handles:
            raise ValueError('No supported large-model attention modules found')
        return self


def build_model(assets, model_name, config, seed, mode, device):
    assert mode == 'bounded'
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    recipe = config['models'][model_name]
    dtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16}[recipe['master_dtype']]
    backbone = AutoModel.from_pretrained(assets / model_name, local_files_only=True,
                                         torch_dtype=dtype, attn_implementation='eager')
    backbone.config.use_cache = False
    backbone.config.problem_type = 'single_label_classification'
    model = make_decoder_sequence_classifier(torch, nn, SequenceClassifierOutput, backbone, 2)
    applied = apply_learnable_frobenius_caps(model, config['cap'], norm_dtype='float32')
    adapters = apply_attention_scaling(model, 'alpha_s_alpha_w', alpha_s=recipe['alpha_s'],
                                      alpha_w=config['cap'], trainable=True,
                                      enforce_alpha_denominator_bound=True)
    configure_trainable(model, recipe['train_scope'], capped=True, trainable_attention_scaling=True)
    # Small trainable tensors remain float32 even with a frozen bfloat16 backbone.
    if dtype == torch.bfloat16:
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.float()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    return model.to(device), applied, adapters


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', choices=['olmo1b', 'mistral7b'], required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--steps', type=int)
    parser.add_argument('--eval-every', type=int)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    core.build_model = build_model
    core.AttentionInputMonitor = LargeAttentionInputMonitor
    core.run(argparse.Namespace(**vars(args), mode='bounded', compile=False))
    path = args.output / 'manifest.json'
    metadata = json.loads(path.read_text())
    recipe = metadata['config']['models'][args.model]
    metadata.update(dtype=recipe['master_dtype'] + ' backbone; float32 trainable tensors; bfloat16 autocast',
                    train_scope=recipe['train_scope'], gradient_checkpointing=True,
                    seed_selection_policy='Exploratory search; all attempts retained, selected successes disclosed.')
    metadata['source_sha256'][str(Path(__file__).resolve())] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    path.write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    main()
