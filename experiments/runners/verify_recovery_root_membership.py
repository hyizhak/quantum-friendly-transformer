"""Independently reconstruct training-root membership from the pinned Parquet files."""

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=Path, default=Path('/tmp/qft-recovery-search-assets'))
    parser.add_argument('--root-parquet', type=Path,
                        default=Path('/tmp/qft-sst-root-assets/default/train/0000.parquet'))
    parser.add_argument('--alignment', type=Path,
                        default=Path('experiments/configs/reviewer/sst_root_training_indices.json'))
    parser.add_argument('--output', type=Path,
                        default=Path('results/reviewer/root_membership_reconstruction.json'))
    args = parser.parse_args()
    alignment = json.loads(args.alignment.read_text())
    train_path = args.assets / 'data/sst2/train-00000-of-00001.parquet'
    validation_path = args.assets / 'data/sst2/validation-00000-of-00001.parquet'
    assert hashlib.sha256(train_path.read_bytes()).hexdigest() == alignment['glue_training_sha256']
    assert hashlib.sha256(args.root_parquet.read_bytes()).hexdigest() == alignment['source']['sha256']
    normalize = lambda s: ' '.join(s.casefold().split())
    roots = {normalize(row['sentence']) for row in pq.read_table(args.root_parquet, columns=['sentence']).to_pylist()}
    train = pq.read_table(train_path).to_pylist()
    validation = pq.read_table(validation_path, columns=['sentence']).to_pylist()
    selected = [row for row in train if normalize(row['sentence']) in roots]
    ids = [row['idx'] for row in selected]
    assert ids == alignment['glue_training_example_ids']
    assert len(ids) == alignment['selected_rows']
    selected_texts = {normalize(row['sentence']) for row in selected}
    assert len(selected_texts) == alignment['selected_unique_texts']
    overlap = selected_texts & {normalize(row['sentence']) for row in validation}
    assert len(overlap) == alignment['validation_text_overlap'] == 0
    for rows, key in [(train, 'unweighted_mean_words'), (selected, 'root_mean_words'),
                      (validation, 'validation_mean_words')]:
        mean = sum(len(row['sentence'].split()) for row in rows) / len(rows)
        assert abs(mean - alignment[key]) < 1e-10
    report = {'source_checksums_match': True, 'exact_training_ids_match': True,
              'selected_training_rows': len(ids), 'selected_unique_texts': len(selected_texts),
              'validation_text_overlap': len(overlap), 'external_labels_read': False,
              'alignment_sha256': hashlib.sha256(args.alignment.read_bytes()).hexdigest()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
