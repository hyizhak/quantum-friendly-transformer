"""Continue a bounded checkpoint on complete training sentences with fresh optimizer state."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

import torch
from experiments.lib import recovery_root_data as root_data
from experiments.lib.recovery_extension import validate_scale_records
from experiments.runners import reviewer_recovery_extension as core
from experiments.runners import reviewer_recovery_large as large
from experiments.runners import reviewer_recovery_matched as matched
from quantum_friendly_transformer.norm_transformer import collect_attention_scaling_stats


def load_parent_state(model, config, name):
    reference = config['parent_checkpoint']
    path = Path(reference['path'])
    assert hashlib.sha256(path.read_bytes()).hexdigest() == reference['sha256']
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    assert checkpoint['mode'] == 'bounded' and checkpoint['model'] == name
    prior = checkpoint['config']
    assert prior['cap'] == config['cap']
    for key in ['model_id', 'revision', 'alpha_s', 'baseline_target']:
        assert prior['models'][name][key] == config['models'][name][key], key
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    validate_scale_records(collect_attention_scaling_stats(model), config['models'][name]['alpha_s'], config['cap'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', choices=['2layers33m', 'olmo1b'], required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--device', required=True)
    parser.add_argument('--steps', type=int)
    parser.add_argument('--eval-every', type=int)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    base_builder = large.build_model if args.model == 'olmo1b' else core.build_model

    def builder(assets, name, config, seed, mode, device):
        model, caps, adapters = base_builder(assets, name, config, seed, mode, device)
        load_parent_state(model, config, name)
        return model, caps, adapters

    core.build_model = builder
    core.prepare_data = root_data.prepare_data
    if args.model == 'olmo1b':
        core.AttentionInputMonitor = large.LargeAttentionInputMonitor
    core.run(argparse.Namespace(**vars(args), mode='bounded', compile=args.model == '2layers33m'))
    path = args.output / 'manifest.json'
    metadata = json.loads(path.read_text())
    for source in [Path(__file__), Path(root_data.__file__), root_data.ALIGNMENT,
                   Path(matched.__file__), Path(large.__file__)]:
        metadata['source_sha256'][str(source.resolve())] = hashlib.sha256(source.read_bytes()).hexdigest()
    metadata.update(selection_protocol='official_validation', training_sampling_protocol='training_roots',
                    initialization='previous bounded checkpoint; fresh optimizer',
                    parent_checkpoint=metadata['config']['parent_checkpoint'])
    metadata['limitations'] = [x for x in metadata['limitations'] if not x.startswith('SST-2 validation is')]
    metadata['limitations'].append('This is continued optimization of a selected bounded parent checkpoint on training roots, not an independent training seed. Validation selects the parent and new checkpoint; the score is not an independent test result.')
    path.write_text(json.dumps(metadata, indent=2) + '\n')
    path = args.output / 'result.json'
    result = json.loads(path.read_text())
    result.update(selection_protocol='official_validation', training_sampling_protocol='training_roots',
                  initialization='previous bounded checkpoint; fresh optimizer',
                  parent_checkpoint=metadata['parent_checkpoint'])
    path.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
