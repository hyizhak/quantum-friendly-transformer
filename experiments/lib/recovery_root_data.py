"""Select complete sentences already present in the pinned SST-2 training set."""

import hashlib
import json
from pathlib import Path
import torch

from experiments.lib.recovery_extension import indices_hash
from experiments.runners import reviewer_recovery_matched as matched

ROOT = Path(__file__).resolve().parents[2]
ALIGNMENT = ROOT / 'experiments/configs/reviewer/sst_root_training_indices.json'


def restrict_to_roots(datasets, metadata, alignment):
    ids = set(alignment['glue_training_example_ids'])
    train = datasets['train']
    keep = torch.tensor([int(i) in ids for i in train['example_id']], dtype=torch.bool)
    assert int(keep.sum()) == alignment['selected_rows']
    datasets['train'] = {key: value[keep] for key, value in train.items()}
    assert set(datasets['train']['example_id'].tolist()) == ids
    size = int(keep.sum())
    metadata['counts']['train'] = size
    metadata['training_original_example_ids'] = datasets['train']['example_id'].tolist()
    metadata['visible_length_mean']['train'] = float(datasets['train']['attention_mask'].sum(-1).float().mean())
    metadata['indices']['train'] = list(range(size))
    metadata['indices']['selection'] = list(range(size, size + len(datasets['selection']['labels'])))
    metadata['sha256'] = {key: indices_hash(value) for key, value in metadata['indices'].items()}
    metadata['index_namespace'] = 'matched original training-root rows, then official validation rows'
    metadata['training_sampling_protocol'] = 'exactly matched complete training sentences only'
    metadata['root_membership_provenance'] = {k: v for k, v in alignment.items() if k != 'glue_training_example_ids'}
    return datasets, metadata


def prepare_data(assets, model_name, config):
    alignment = json.loads(ALIGNMENT.read_text())
    manifest = json.loads((assets / 'assets.json').read_text())
    assert manifest['sha256']['data/sst2/train-00000-of-00001.parquet'] == alignment['glue_training_sha256']
    assert config['dataset_revision'] == alignment['glue_dataset_revision']
    tokenizer, datasets, metadata = matched.prepare_data(assets, model_name, config)
    datasets, metadata = restrict_to_roots(datasets, metadata, alignment)
    metadata['root_alignment_sha256'] = hashlib.sha256(ALIGNMENT.read_bytes()).hexdigest()
    return tokenizer, datasets, metadata
