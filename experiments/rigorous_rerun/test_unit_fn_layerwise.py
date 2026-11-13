"""Lock and test both completed layerwise-distillation variants."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import time


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def lock(root):
    plan = json.loads((root / 'plan.json').read_text())
    assert sha(root / 'unit_fn_layerwise_distill.py') == plan['runner_sha256']
    candidates = []
    for trial in plan['trials']:
        folder = root / trial['name']
        record = json.loads((folder / 'selected.json').read_text())
        assert sha(folder / 'best.pt') == record['checkpoint_sha256']
        history = json.loads((folder / 'history.json').read_text())
        best = max(history, key=lambda x: x['validation']['accuracy'])
        assert best['epoch'] == record['selected_epoch']
        assert best['validation'] == record['validation']
        candidates.append(record)
    data = {'plan_sha256': sha(root / 'plan.json'), 'test_script_sha256': sha(Path(__file__)),
            'candidates': candidates,
            'validation_selected_variant': max(candidates, key=lambda x: x['validation']['accuracy'])['trial']['name'],
            'rule': 'Each variant uses its highest-validation-accuracy checkpoint; report both complete test results.',
            'prior_test_disclosure': plan['prior_test_disclosure']}
    output = root / 'test_results'
    output.mkdir(exist_ok=True)
    path = output / 'selection_lock.json'
    if path.exists():
        assert json.loads(path.read_text()) == data
    else:
        write(path, data)
    print('Both checkpoint selections locked before test evaluation.', flush=True)


def evaluate(root, worker_id):
    import numpy as np
    import torch

    output = root / 'test_results'
    locked = json.loads((output / 'selection_lock.json').read_text())
    assert sha(Path(__file__)) == locked['test_script_sha256']
    assert sha(root / 'plan.json') == locked['plan_sha256']
    plan = json.loads((root / 'plan.json').read_text())
    runner = root / 'unit_fn_layerwise_distill.py'
    assert sha(runner) == plan['runner_sha256']
    spec = importlib.util.spec_from_file_location('frozen_layerwise', runner)
    training = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(training)
    assert sha(training.RECOVERY) == plan['recovery_dependency_sha256']
    assert sha(training.PREVIOUS / 'plan.json') == plan['previous_plan_sha256']
    training.base.configure()
    selected = next(x for x in locked['candidates'] if x['trial']['worker'] == worker_id)
    trial = selected['trial']
    folder = output / trial['name']
    folder.mkdir(exist_ok=True)
    checkpoint = root / trial['name'] / 'best.pt'
    assert sha(checkpoint) == selected['checkpoint_sha256']
    if (folder / 'result.json').exists():
        report = json.loads((folder / 'result.json').read_text())
        assert report['predictions_sha256'] == sha(folder / 'test_predictions.npz')
        assert report['selection_lock_sha256'] == sha(output / 'selection_lock.json')
        return
    model, _ = training.build_student(plan, trial)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict(state['model'], strict=True)
    del state
    original = Path(plan['original_study'])
    for split in ('validation', 'test'):
        assert sha(original / 'data/genomic' / f'{split}.pt') == plan['data_sha256'][split]
    validation = torch.load(original / 'data/genomic/validation.pt', weights_only=True)
    metrics, predictions, _ = training.base.evaluate(model, validation, training.search.JOB, trial['eval_batch_size'])
    with np.load(root / trial['name'] / 'validation_predictions.npz') as saved:
        assert np.array_equal(predictions.numpy(), saved['predictions'])
    assert metrics['accuracy'] == selected['validation']['accuracy']
    norms = training.search.audit(model)
    test = torch.load(original / 'data/genomic/test.pt', weights_only=True)
    metrics, predictions, logits = training.base.evaluate(model, test, training.search.JOB,
                                                         trial['eval_batch_size'], return_logits=True)
    y, p = test['labels'].numpy(), predictions.numpy()
    assert len(y) == 9034 and logits.shape == (9034, 2) and torch.isfinite(logits).all()
    counts = {'tp': int(((y == 1) & (p == 1)).sum()), 'tn': int(((y == 0) & (p == 0)).sum()),
              'fp': int(((y == 0) & (p == 1)).sum()), 'fn': int(((y == 1) & (p == 0)).sum())}
    assert sum(counts.values()) == 9034
    assert metrics['accuracy'] == (counts['tp'] + counts['tn']) / 9034
    assert abs(metrics['f1'] - 2 * counts['tp'] / (2 * counts['tp'] + counts['fp'] + counts['fn'])) < 1e-12
    np.savez_compressed(folder / 'test_predictions.npz', labels=y, predictions=p, logits=logits.numpy())
    report = {'trial': trial, 'selected_epoch': selected['selected_epoch'], 'trained_epochs': selected['trained_epochs'],
              'validation': selected['validation'], 'test': metrics, 'confusion_counts': counts, 'norms': norms,
              'checkpoint_sha256': sha(checkpoint), 'selection_lock_sha256': sha(output / 'selection_lock.json'),
              'predictions_sha256': sha(folder / 'test_predictions.npz'),
              'prior_test_disclosure': plan['prior_test_disclosure'],
              'completed_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    write(folder / 'result.json', report)
    print(json.dumps({'variant': trial['name'], 'test': metrics}), flush=True)


def assemble(root):
    import numpy as np

    output = root / 'test_results'
    plan = json.loads((root / 'plan.json').read_text())
    locked = json.loads((output / 'selection_lock.json').read_text())
    reports = []
    labels = None
    for trial in plan['trials']:
        folder = output / trial['name']
        report = json.loads((folder / 'result.json').read_text())
        assert report['selection_lock_sha256'] == sha(output / 'selection_lock.json')
        assert report['predictions_sha256'] == sha(folder / 'test_predictions.npz')
        assert report['checkpoint_sha256'] == sha(root / trial['name'] / 'best.pt')
        with np.load(folder / 'test_predictions.npz') as data:
            y, p, logits = data['labels'], data['predictions'], data['logits']
            assert np.array_equal(p, logits.argmax(-1)) and np.isfinite(logits).all()
            if labels is not None:
                assert np.array_equal(y, labels)
            labels = y.copy()
            assert len(y) == 9034 and float((y == p).mean()) == report['test']['accuracy']
            tp, fp, fn = ((y == 1) & (p == 1)).sum(), ((y == 0) & (p == 1)).sum(), ((y == 1) & (p == 0)).sum()
            assert abs(float(2 * tp / (2 * tp + fp + fn)) - report['test']['f1']) < 1e-12
        assert len(report['norms']) == 72 and max(x['frobenius_fp64'] for x in report['norms']) <= 1
        reports.append(report)
    repository = Path(plan['repository'])
    summary = {'validation_selected_variant': locked['validation_selected_variant'], 'seed': 42,
               'reports': reports, 'test_examples': 9034, 'independent_prediction_recount_passed': True,
               'all_144_effective_matrix_norms_at_most_one': True}
    write(output / 'summary.json', summary)
    exported = repository / 'artifacts/rigorous_rerun/unit_fn_layerwise_test'
    shutil.copytree(output, exported, dirs_exist_ok=True)
    shutil.copy2(root / 'plan.json', exported / 'plan.json')
    shutil.copy2(Path(__file__), exported / Path(__file__).name)
    for trial in plan['trials']:
        for name in ('history.json', 'selected.json', 'initial_audit.json', 'validation_predictions.npz'):
            shutil.copy2(root / trial['name'] / name, exported / trial['name'] / name)
    write(exported / 'complete.json', {'summary': str(exported / 'summary.json'),
                                      'test_variants': len(reports)})
    print(json.dumps({'complete': True, 'results': [
        {'variant': r['trial']['name'], 'validation': r['validation'], 'test': r['test']} for r in reports]}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['lock', 'evaluate', 'assemble'])
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--worker', type=int, default=0)
    args = parser.parse_args()
    if args.action == 'lock':
        lock(args.root.resolve())
    elif args.action == 'evaluate':
        evaluate(args.root.resolve(), args.worker)
    else:
        assemble(args.root.resolve())
