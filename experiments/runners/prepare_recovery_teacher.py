"""Cache a pinned external SST-2 teacher on the original training rows only."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

import pyarrow.parquet as pq
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from experiments.lib.recovery_full_dense import file_sha256
from experiments.lib.experiment import manifest, save_json

TEACHER_ID = 'textattack/roberta-base-SST-2'
TEACHER_REVISION = '84ee248b91053ef5d0c748bbac4edfba1cf89584'


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_assets, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.teacher_assets,
        local_files_only=True, dtype=torch.float32, attn_implementation='eager').to(device).eval()
    model.requires_grad_(False)
    rows_by_split = {
        'train': pq.read_table(args.assets / 'data/sst2/train-00000-of-00001.parquet').to_pylist(),
        'validation': pq.read_table(args.assets / 'data/sst2/validation-00000-of-00001.parquet').to_pylist(),
    }
    started = time.monotonic()
    metrics = {}
    for split, rows in rows_by_split.items():
        chunks = []
        for start in range(0, len(rows), args.batch_size):
            texts = [r['sentence'] for r in rows[start:start + args.batch_size]]
            batch = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                logits = model(**batch).logits.float().cpu()
            assert logits.shape == (len(texts), 2) and bool(torch.isfinite(logits).all())
            chunks.append(logits)
            if start % (args.batch_size * 25) == 0:
                print(f'{split} {start}/{len(rows)} elapsed={time.monotonic()-started:.1f}s', flush=True)
        logits = torch.cat(chunks)
        labels = torch.tensor([r['label'] for r in rows])
        ids = torch.tensor([r['idx'] for r in rows])
        metrics[split] = {'correct': int((logits.argmax(-1) == labels).sum()), 'examples': len(labels),
                          'accuracy': float((logits.argmax(-1) == labels).double().mean()),
                          'mean_nll': float(torch.nn.functional.cross_entropy(logits, labels))}
        if split == 'train':
            # Validation logits are deliberately absent from the training cache.
            torch.save({'logits': logits, 'example_id': ids, 'labels': labels,
                        'text_sha256': [hashlib.sha256(r['sentence'].encode()).hexdigest() for r in rows],
                        'teacher_id': TEACHER_ID, 'teacher_revision': TEACHER_REVISION,
                        'split': 'train'}, args.output / 'training_logits.pth')
        else:
            from experiments.lib.experiment import save_csv
            save_csv(args.output / 'teacher_validation_predictions.csv', [
                {'example_id': int(i), 'label': int(y), 'prediction': int(p)}
                for i, y, p in zip(ids, labels, logits.argmax(-1))])
    assert metrics['validation']['accuracy'] > .90, 'Check teacher class mapping or checkpoint quality'
    asset_hashes = {p.name: file_sha256(p) for p in args.teacher_assets.iterdir() if p.is_file()}
    payload = manifest({}, [Path(__file__), ROOT / 'experiments/lib/recovery_full_dense.py',
                           ROOT / 'experiments/lib/experiment.py'], teacher_id=TEACHER_ID,
        teacher_revision=TEACHER_REVISION, teacher_assets_sha256=asset_hashes,
        training_logits_sha256=file_sha256(args.output / 'training_logits.pth'),
        dataset_id='nyu-mll/glue', dataset_revision='bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c',
        training_data_sha256=file_sha256(args.assets / 'data/sst2/train-00000-of-00001.parquet'),
        label_order=['negative', 'positive'], sequence_length=128,
        teacher_evaluation=metrics, seconds=time.monotonic()-started,
        protocol='External SST-2-finetuned teacher; only original training texts have cached distillation targets. Validation is evaluated separately and never supplied to the student loss.',
        limitations=['Teacher fine-tuning provenance is the published external checkpoint, not our own controlled training run.',
                     'This adds an external teacher to the training resources; do not conflate it with the earlier label-only recovery recipe.'],
        sources=['https://huggingface.co/textattack/roberta-base-SST-2/tree/' + TEACHER_REVISION,
                 'https://arxiv.org/abs/1503.02531'])
    save_json(args.output / 'manifest.json', payload)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--teacher-assets', type=Path, required=True)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--batch-size', type=int, default=128)
    main(parser.parse_args())
