"""Run one transparently selected-seed recovery attempt for any TinyStories model."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    from experiments.runners.recovery_extension import run
    run(argparse.Namespace(**vars(args), mode='bounded', compile=True,
                           steps=None, eval_every=None, smoke=False))
    path = args.output / 'manifest.json'
    metadata = json.loads(path.read_text())
    metadata['source_sha256'][str(Path(__file__).resolve())] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    metadata['seed_selection_policy'] = (
        'Exploratory seed search; all attempts retained. A qualifying validation result '
        'can stop further seeds. Selected successes do not estimate average-seed performance.')
    path.write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    main()
