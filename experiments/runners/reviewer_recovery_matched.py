"""Match the archived full-training/validation-selected recovery protocol.

These are explicitly selected validation examples, not independent test scores.
The actual training loop and bounded attention implementation are unchanged.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

import torch
from experiments.lib.recovery_extension import indices_hash
from experiments.runners import reviewer_recovery_extension as core

original_prepare_data = core.prepare_data


def prepare_data(assets, model_name, config):
    tokenizer, datasets, original = original_prepare_data(assets, model_name, config)
    combined = {key: torch.cat([datasets['train'][key], datasets['selection'][key]])
                for key in datasets['train']}
    order = combined['example_id'].argsort()
    combined = {key: values[order] for key, values in combined.items()}
    assert len(combined['example_id'].unique()) == len(combined['example_id'])
    evaluation = datasets['evaluation']
    datasets = {'train': combined, 'selection': evaluation, 'evaluation': evaluation}
    size = len(combined['labels'])
    # One unambiguous index namespace: retained training rows followed by
    # official validation rows. No validation row is included in training.
    indices = {'train': list(range(size)),
               'selection': list(range(size, size + len(evaluation['labels']))),
               'excluded_evaluation_overlap': original['indices']['excluded_evaluation_overlap']}
    metadata = {'indices': indices, 'sha256': {k: indices_hash(v) for k, v in indices.items()},
                'index_namespace': 'retained original training rows, then official validation rows',
                'training_original_example_ids': combined['example_id'].tolist(),
                'selection_original_example_ids': evaluation['example_id'].tolist(),
                'selection_equals_evaluation': True,
                'selection_protocol': 'official_validation',
                'counts': {name: len(rows['labels']) for name, rows in datasets.items()},
                'visible_length_mean': {name: float(rows['attention_mask'].sum(-1).float().mean())
                                        for name, rows in datasets.items()}}
    return tokenizer, datasets, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', choices=['33m', '2layers33m'], required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--device', required=True)
    args = parser.parse_args()
    core.prepare_data = prepare_data
    core.run(argparse.Namespace(**vars(args), mode='bounded', compile=True,
                               steps=None, eval_every=None, smoke=False))
    path = args.output / 'manifest.json'
    metadata = json.loads(path.read_text())
    metadata['source_sha256'][str(Path(__file__).resolve())] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    metadata['selection_protocol'] = 'official_validation'
    metadata['limitations'] = [x for x in metadata['limitations'] if not x.startswith('SST-2 validation is')]
    metadata['limitations'].append('Official validation selects checkpoints and seeds. These are selected recovery examples, not independent test scores. Full retained SST-2 training data is used, matching the archived data/selection policy.')
    path.write_text(json.dumps(metadata, indent=2) + '\n')
    path = args.output / 'result.json'
    result = json.loads(path.read_text())
    result['selection_protocol'] = 'official_validation'
    path.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
