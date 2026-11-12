"""Train bounded low-rank Mistral recovery without relaxing either cap."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

from experiments.lib.bounded_low_rank import install_bounded_low_rank
from experiments.runners import recovery_extension as core
from experiments.runners import recovery_large as large


def build_model(assets, model_name, config, seed, mode, device):
    model, applied, adapters = large.build_model(assets, model_name, config, seed, mode, device)
    recipe = config['models'][model_name]
    count = install_bounded_low_rank(model, recipe['low_rank_rank'], recipe['low_rank_alpha'])
    assert count == applied
    return model, applied, adapters


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--steps', type=int)
    parser.add_argument('--eval-every', type=int)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    core.build_model = build_model
    core.AttentionInputMonitor = large.LargeAttentionInputMonitor
    config = json.loads(args.config.read_text())
    root_sources = []
    if config.get('training_sampling_protocol') == 'training_roots':
        from experiments.lib import recovery_root_data as root_data
        from experiments.runners import recovery_matched as matched
        core.prepare_data = root_data.prepare_data
        root_sources = [Path(root_data.__file__), root_data.ALIGNMENT, Path(matched.__file__)]
    core.run(argparse.Namespace(**vars(args), model='mistral7b', mode='bounded', compile=False))
    path = args.output / 'manifest.json'
    metadata = json.loads(path.read_text())
    metadata.update(dtype='bfloat16 frozen base weights; float32 low-rank/head/norm/cap parameters; bfloat16 autocast',
                    train_scope='bounded low-rank weight updates plus head/norm/caps',
                    composed_weight_cap=True, gradient_checkpointing=True)
    for source in [Path(__file__), Path(large.__file__), ROOT / 'experiments/lib/bounded_low_rank.py'] + root_sources:
        metadata['source_sha256'][str(source.resolve())] = hashlib.sha256(source.read_bytes()).hexdigest()
    metadata['limitations'].append('Low-rank updates are composed before the Frobenius cap. This changes trainable scope from the archived head_caps recipe; the inference architecture and configured attention floor remain unchanged.')
    if root_sources:
        metadata.update(selection_protocol='official_validation', training_sampling_protocol='training_roots')
        metadata['limitations'] = [x for x in metadata['limitations'] if not x.startswith('SST-2 validation is')]
        metadata['limitations'].append('Only exact full-sentence matches from the original SST-2 training set are used. Official validation selects checkpoints; scores are selected examples, not independent test results.')
    path.write_text(json.dumps(metadata, indent=2) + '\n')
    if root_sources:
        path = args.output / 'result.json'
        result = json.loads(path.read_text())
        result.update(selection_protocol='official_validation', training_sampling_protocol='training_roots')
        path.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
