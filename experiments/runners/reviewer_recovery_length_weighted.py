"""Emphasize longer training phrases without using validation labels to train."""

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
from experiments.runners import reviewer_recovery_matched as matched


def prepare_data(assets, model_name, config):
    tokenizer, datasets, metadata = matched.prepare_data(assets, model_name, config)
    original = datasets['train']
    lengths = original['attention_mask'].sum(-1)
    repetitions = ((lengths + 11) // 12).clamp(min=1, max=6)
    order = torch.arange(len(lengths)).repeat_interleave(repetitions)
    datasets['train'] = {key: value[order] for key, value in original.items()}
    size = len(order)
    metadata['unweighted_training_count'] = len(lengths)
    metadata['unweighted_training_visible_length_mean'] = float(lengths.float().mean())
    metadata['repetitions_per_training_example'] = repetitions.tolist()
    metadata['training_original_example_ids'] = datasets['train']['example_id'].tolist()
    metadata['counts']['train'] = size
    metadata['visible_length_mean']['train'] = float(lengths[order].float().mean())
    metadata['indices']['train'] = list(range(size))
    metadata['indices']['selection'] = list(range(size, size + len(datasets['selection']['labels'])))
    metadata['index_namespace'] = 'length-weighted training rows, then official validation rows'
    metadata['sha256'] = {key: indices_hash(value) for key, value in metadata['indices'].items()}
    metadata['training_sampling_protocol'] = 'repeat each training example min(6, ceil(visible_tokens/12)) times'
    return tokenizer, datasets, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', choices=['2layers33m'], required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--device', required=True)
    args = parser.parse_args()
    core.prepare_data = prepare_data
    core.run(argparse.Namespace(**vars(args), mode='bounded', compile=True, steps=None, eval_every=None, smoke=False))
    path = args.output / 'manifest.json'
    metadata = json.loads(path.read_text())
    for source in [Path(__file__), Path(matched.__file__)]:
        metadata['source_sha256'][str(source.resolve())] = hashlib.sha256(source.read_bytes()).hexdigest()
    metadata.update(selection_protocol='official_validation', training_sampling_protocol='length_weighted')
    metadata['limitations'] = [x for x in metadata['limitations'] if not x.startswith('SST-2 validation is')]
    metadata['limitations'].append('Official validation selects checkpoints and seeds; these are selected examples, not independent test scores. Training-example repetition depends only on training token lengths.')
    path.write_text(json.dumps(metadata, indent=2) + '\n')
    path = args.output / 'result.json'
    result = json.loads(path.read_text())
    result.update(selection_protocol='official_validation', training_sampling_protocol='length_weighted')
    path.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
