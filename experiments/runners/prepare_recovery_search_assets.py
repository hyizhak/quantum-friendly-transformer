"""Download pinned public checkpoints, including sharded OLMo and Mistral."""

import argparse
import hashlib
import json
from pathlib import Path
from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    for name, model in config['models'].items():
        if name in ['1m', '8m', '33m']:
            original = Path('/tmp/qft-recovery-assets') / name
        elif name == '3m':
            original = Path('/tmp/qft-recovery-positive-control-assets') / name
        else:
            original = None
        destination = args.output / name
        if original is not None and original.exists():
            if not destination.exists():
                destination.symlink_to(original, target_is_directory=True)
        else:
            weights = ['pytorch_model.bin'] if name in ['28m', '1layer21m', '2layers33m'] else ['model*.safetensors', 'model.safetensors.index.json']
            snapshot_download(model['model_id'], revision=model['revision'], token=False,
                              allow_patterns=weights + ['config.json', 'tokenizer.json', 'tokenizer.model',
                              'tokenizer_config.json', 'special_tokens_map.json', 'vocab.json', 'merges.txt'],
                              local_dir=destination, max_workers=2)
        print(f'Prepared {name}', flush=True)
    data = args.output / 'data'
    if not data.exists():
        data.symlink_to('/tmp/qft-recovery-assets/data', target_is_directory=True)
    files = {}
    for directory in [args.output / name for name in config['models']] + [data]:
        for path in sorted(directory.rglob('*')):
            if path.is_file() and '.cache' not in path.parts:
                with path.open('rb') as handle:
                    files[str(path.relative_to(args.output))] = hashlib.file_digest(handle, 'sha256').hexdigest()
    record = {'dataset_id': config['dataset_id'], 'dataset_revision': config['dataset_revision'],
              'models': {n: {k: m[k] for k in ['model_id', 'revision']} for n, m in config['models'].items()},
              'sha256': files}
    (args.output / 'assets.json').write_text(json.dumps(record, indent=2) + '\n')
    print(f'Hashed {len(files)} files', flush=True)


if __name__ == '__main__':
    main()
