"""Adapt only the existing classifier on training sentences; preserve the bounded backbone."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

from experiments.lib import recovery_root_data as root_data
from experiments.runners import recovery_extension as core
from experiments.runners import recovery_large as large
from experiments.runners import recovery_low_rank as low_rank
from experiments.runners import recovery_curriculum as curriculum
from experiments.runners import recovery_matched as matched

original_builder = core.build_model


def build_model(assets, name, config, seed, mode, device):
    recipe = config['models'][name]
    builder = low_rank.build_model if 'low_rank_rank' in recipe else (
        large.build_model if name in {'olmo1b', 'mistral7b'} else original_builder)
    model, caps, adapters = builder(assets, name, config, seed, mode, device)
    curriculum.load_parent_state(model, config, name)
    model.requires_grad_(False)
    model.score.requires_grad_(True)
    return model, caps, adapters


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint-root', type=Path)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', choices=['2layers33m', 'olmo1b', 'mistral7b'], required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--device', required=True)
    args = parser.parse_args()
    core.build_model = build_model
    core.prepare_data = root_data.prepare_data
    if args.model != '2layers33m':
        core.AttentionInputMonitor = large.LargeAttentionInputMonitor
    core.run(argparse.Namespace(**vars(args), mode='bounded', compile=args.model == '2layers33m',
                               steps=None, eval_every=None, smoke=False))
    path = args.output / 'manifest.json'
    metadata = json.loads(path.read_text())
    for source in [Path(__file__), Path(root_data.__file__), root_data.ALIGNMENT,
                   Path(matched.__file__), Path(large.__file__), Path(low_rank.__file__),
                   Path(curriculum.__file__), ROOT / 'experiments/lib/bounded_low_rank.py']:
        metadata['source_sha256'][str(source.resolve())] = hashlib.sha256(source.read_bytes()).hexdigest()
    metadata.update(selection_protocol='official_validation', training_sampling_protocol='training_roots',
                    initialization='previous bounded checkpoint; classifier-only adaptation',
                    dtype='Backbone tensor dtypes inherited exactly from bounded parent; float32 trainable classifier; bfloat16 autocast',
                    parent_checkpoint=metadata['config']['parent_checkpoint'], train_scope='existing classifier only')
    metadata['limitations'] = [x for x in metadata['limitations'] if not x.startswith('SST-2 validation is')]
    metadata['limitations'].append('Only the existing bias-free classifier is trained on training roots. The bounded backbone, caps, and scale controllers are frozen. Validation selects the parent and checkpoint, so this is a selected example rather than an independent test score.')
    path.write_text(json.dumps(metadata, indent=2) + '\n')
    path = args.output / 'result.json'
    result = json.loads(path.read_text())
    result.update(selection_protocol='official_validation', training_sampling_protocol='training_roots',
                  initialization=metadata['initialization'], parent_checkpoint=metadata['parent_checkpoint'])
    path.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
