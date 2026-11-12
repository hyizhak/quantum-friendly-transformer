"""Evaluate single-model parameter averages of compatible bounded checkpoints."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

import torch
from experiments.lib.recovery_extension import validate_scale_records
from experiments.runners import reviewer_recovery_extension as core
from experiments.runners import reviewer_recovery_matched as matched
from quantum_friendly_transformer.norm_transformer import collect_attention_scaling_stats

original_builder = core.build_model


def apply_average(model, config, name, candidate):
    references = config['parent_checkpoints']
    coefficients = config['candidate_coefficients'][str(candidate)]
    assert len(coefficients) == len(references)
    assert all(x >= 0 for x in coefficients) and abs(sum(coefficients) - 1.) < 1e-12
    parameters = set(dict(model.named_parameters()))
    state = None
    for reference, coefficient in zip(references, coefficients):
        path = Path(reference['path'])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == reference['sha256']
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        assert checkpoint['model'] == name and checkpoint['mode'] == 'bounded'
        prior = checkpoint['config']
        assert prior['cap'] == config['cap']
        for key in ['model_id', 'revision', 'alpha_s', 'baseline_target']:
            assert prior['models'][name][key] == config['models'][name][key], key
        incoming = checkpoint['model_state_dict']
        if state is None:
            state = {key: value.clone() * coefficient if key in parameters else value.clone()
                     for key, value in incoming.items()}
        else:
            assert state.keys() == incoming.keys()
            for key, value in incoming.items():
                if key in parameters:
                    state[key].add_(value, alpha=coefficient)
                else:
                    # Encoding bounds and other inference buffers must agree.
                    assert torch.equal(state[key], value), key
    model.load_state_dict(state, strict=True)
    validate_scale_records(collect_attention_scaling_stats(model), config['models'][name]['alpha_s'], config['cap'])


def build_model(assets, name, config, candidate, mode, device):
    model, caps, adapters = original_builder(assets, name, config, candidate, mode, device)
    apply_average(model, config, name, candidate)
    return model, caps, adapters


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--candidate', type=int, required=True)
    parser.add_argument('--device', required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    assert config['models']['2layers33m']['steps'] == 0
    core.build_model = build_model
    core.prepare_data = matched.prepare_data
    core.run(argparse.Namespace(config=args.config, assets=args.assets, output=args.output,
                               model='2layers33m', mode='bounded', seed=args.candidate, device=args.device,
                               compile=False, steps=None, eval_every=None, smoke=False))
    path = args.output / 'manifest.json'
    metadata = json.loads(path.read_text())
    for source in [Path(__file__), Path(matched.__file__)]:
        metadata['source_sha256'][str(source.resolve())] = hashlib.sha256(source.read_bytes()).hexdigest()
    extra = dict(selection_protocol='official_validation', training_sampling_protocol='no_new_training',
                 initialization='single-model parameter average of bounded parents', training_updates=0,
                 seed_role='candidate identifier; no new training seed',
                 parent_checkpoints=config['parent_checkpoints'],
                 averaging_coefficients=config['candidate_coefficients'][str(args.candidate)])
    metadata.update(extra)
    metadata['limitations'] = [x for x in metadata['limitations'] if not x.startswith('SST-2 validation is')]
    metadata['limitations'].append('Only model parameters are averaged; inference buffers agree exactly. This remains one model with the same architecture and enforced caps/floor. Parents and averaging coefficients are selected with validation, so this is a selected example, not an independent test or training seed.')
    path.write_text(json.dumps(metadata, indent=2) + '\n')
    path = args.output / 'result.json'
    result = json.loads(path.read_text())
    result.update(extra)
    path.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
