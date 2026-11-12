"""Bounded recovery on complete SST-2 training sentences, with disclosed validation selection."""

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
from experiments.runners import recovery_matched as matched


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', choices=['2layers33m', 'olmo1b'], required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--device', required=True)
    args = parser.parse_args()
    core.prepare_data = root_data.prepare_data
    if args.model == 'olmo1b':
        core.build_model = large.build_model
        core.AttentionInputMonitor = large.LargeAttentionInputMonitor
    core.run(argparse.Namespace(**vars(args), mode='bounded', compile=args.model == '2layers33m',
                               steps=None, eval_every=None, smoke=False))
    path = args.output / 'manifest.json'
    metadata = json.loads(path.read_text())
    for source in [Path(__file__), Path(root_data.__file__), root_data.ALIGNMENT, Path(matched.__file__), Path(large.__file__)]:
        metadata['source_sha256'][str(source.resolve())] = hashlib.sha256(source.read_bytes()).hexdigest()
    metadata.update(selection_protocol='official_validation', training_sampling_protocol='training_roots')
    metadata['limitations'] = [x for x in metadata['limitations'] if not x.startswith('SST-2 validation is')]
    metadata['limitations'].append('Training uses 6803 exact whole-sentence matches from existing SST-2 training labels. External training-only metadata identifies roots. Official validation selects checkpoints and seeds; scores are selected examples, not independent test scores.')
    path.write_text(json.dumps(metadata, indent=2) + '\n')
    path = args.output / 'result.json'
    result = json.loads(path.read_text())
    result.update(selection_protocol='official_validation', training_sampling_protocol='training_roots')
    path.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
