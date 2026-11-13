"""Seed-42 FN1 follow-up with ordinary trainable LayerNorm.

Two prespecified training objectives share the same ordinary-LN architecture.
Training reads only train/validation; each validation-selected checkpoint is
locked before test. Previous test exposure is disclosed in the manifest.
"""

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import get_cosine_schedule_with_warmup


PREVIOUS = Path('/data/hyizhan/qft-fn1-search-20261113')
RECOVERY = Path('/data/hyizhan/qft-fn1-ffn-recovery-20261113/recover_unit_fn_ffn.py')
spec = importlib.util.spec_from_file_location('frozen_recovery', RECOVERY)
recovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery)
search, base = recovery.search, recovery.base


def prepare(root):
    old, original, study = search.load_plan(PREVIOUS)
    root.mkdir(exist_ok=False)
    repository = Path(old['repository'])
    common = {**old['trials'][0], 'initialization': 'teacher12', 'epochs': 48,
              'min_epochs': 16, 'patience': 10, 'warmup_ratio': 0.05,
              'attention_lr': 2e-5, 'ffn_lr': 1e-4, 'other_lr': 2e-4,
              'adam_epsilon': 1e-12, 'midnorm': False, 'smoothing': 0.0,
              'representation_weight': 0.0, 'representation_warmup_epochs': 0}
    common.pop('lr')
    common.pop('aux_lr')
    trials = [
        {**common, 'name': 'fn1_trainable_ln_kd', 'worker': 0},
        {**common, 'name': 'fn1_trainable_ln_layerwise', 'worker': 1,
         'representation_weight': 0.15, 'representation_warmup_epochs': 2},
    ]
    shutil.copy2(__file__, root / Path(__file__).name)
    plan = {
        'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'runner_sha256': base.file_sha(root / Path(__file__).name),
        'recovery_dependency_sha256': base.file_sha(RECOVERY),
        'search_dependency_sha256': base.file_sha(PREVIOUS / 'unit_fn_search.py'),
        'previous_plan_sha256': base.file_sha(PREVIOUS / 'plan.json'),
        'repository': str(repository), 'original_study': str(original),
        'original_study_sha256': study['study_sha256'],
        'teacher_checkpoint': old['teacher_checkpoint'], 'teacher_sha256': old['teacher_sha256'],
        'teacher_cache_sha256': base.file_sha(PREVIOUS / 'teacher_train.pt'),
        'data_sha256': old['data_sha256'], 'trials': trials,
        'architecture': {
            'layernorm': '25 ordinary nn.LayerNorm modules, with trainable per-coordinate gain and bias initialized from the native teacher. No unit-output constraint.',
            'projection_frobenius_max': 1.0, 'projection_count': 72,
            'scope': 'Complete Q/K/V/attention-output/gated-FFN/FFN-output weights. Embeddings, pooler, classifier, projection biases and LayerNorm affine parameters are outside this weight cap, as disclosed.',
            'attention_score_multiplier': 8.0, 'ffn': 'Original gated GELU with trainable residual scale; no intermediate activation normalization.',
            'initialization': 'Native seed42 DNABERT-2 teacher selected at epoch12 by validation. Projection directions, biases, embeddings, pooler and classifier copied; original FN residual scale retained.',
        },
        'optimization': 'FP32 effective weight normalization; canonical raw weight retraction after each AdamW update; separate FFN/attention/other gradient clipping; BF16 training and FP32 validation/test. Cached training-only logit distillation retained. Second trial adds centered unit-direction layerwise distillation from a frozen training-only teacher.',
        'selection': 'Highest validation accuracy, first checkpoint wins ties, including epoch zero. Variant selection also uses validation accuracy. Each candidate gets its own locked validation-best test evaluation; test never selects a checkpoint or variant.',
        'prior_test_disclosure': 'Earlier FN1 test results have already been observed. These are exploratory follow-up results on the same split, not a fresh confirmatory test.',
        'theory_scope': 'Empirical weight-FN1 experiment with ordinary trainable LayerNorm, not the fixed-unit-LayerNorm theorem architecture.',
    }
    base.write_json(root / 'plan.json', plan)
    base.write_json(repository / 'artifacts/rigorous_rerun/trainable_ln_fn1_plan.json', plan)
    print(json.dumps({'prepared': str(root), 'trials': trials}), flush=True)


def verify(root):
    plan = json.loads((root / 'plan.json').read_text())
    assert base.file_sha(__file__) == plan['runner_sha256']
    assert base.file_sha(RECOVERY) == plan['recovery_dependency_sha256']
    assert base.file_sha(PREVIOUS / 'unit_fn_search.py') == plan['search_dependency_sha256']
    assert base.file_sha(PREVIOUS / 'plan.json') == plan['previous_plan_sha256']
    assert base.file_sha(Path(plan['teacher_checkpoint'])) == plan['teacher_sha256']
    return plan


def build(plan, trial):
    old, original, study = search.load_plan(PREVIOUS)
    model = search.build(old, original, study, trial)
    teacher_state = torch.load(plan['teacher_checkpoint'], map_location='cpu', weights_only=True)['model']
    replaced = []
    for name, module in list(model.network.named_modules()):
        if isinstance(module, base.UnitNormLayerNorm):
            ordinary = nn.LayerNorm(module.normalized_shape, eps=module.eps)
            ordinary.load_state_dict({k: teacher_state[f'{name}.{k}'] for k in ('weight', 'bias')}, strict=True)
            parent_name, _, leaf = name.rpartition('.')
            parent = model.network.get_submodule(parent_name) if parent_name else model.network
            setattr(parent, leaf, ordinary)
            replaced.append(name)
    assert len(replaced) == 25
    assert not any(isinstance(m, base.UnitNormLayerNorm) for m in model.modules())
    model.cuda()
    recovery.retract(model)
    return model, replaced


def affine_audit(model):
    rows = []
    for name, module in model.named_modules():
        if isinstance(module, nn.LayerNorm):
            assert module.weight.requires_grad and module.bias.requires_grad
            assert torch.isfinite(module.weight).all() and torch.isfinite(module.bias).all()
            rows.append({'module': name, 'width': module.weight.numel(), 'epsilon': module.eps,
                         'gain_min': module.weight.min().item(), 'gain_max': module.weight.max().item(),
                         'gain_l2': module.weight.norm().item(), 'bias_l2': module.bias.norm().item(),
                         'trainable': True})
    assert len(rows) == 25
    return rows


def train(root, worker_id):
    base.configure()
    plan = verify(root)
    original = Path(plan['original_study'])
    trial = next(t for t in plan['trials'] if t['worker'] == worker_id)
    folder = root / trial['name']
    folder.mkdir(exist_ok=False)
    base.seed_everything(42)
    model, replaced = build(plan, trial)
    for split in ('train', 'validation'):
        assert base.file_sha(original / 'data/genomic' / f'{split}.pt') == plan['data_sha256'][split]
    train_data = torch.load(original / 'data/genomic/train.pt', weights_only=True)
    validation = torch.load(original / 'data/genomic/validation.pt', weights_only=True)
    assert base.file_sha(PREVIOUS / 'teacher_train.pt') == plan['teacher_cache_sha256']
    cache = torch.load(PREVIOUS / 'teacher_train.pt', weights_only=True)
    teacher, hooks = None, []
    student_states, teacher_states = [], []
    if trial['representation_weight']:
        study = json.loads((original / 'study.json').read_text())
        teacher = base.baseline_multilayer(study)
        state = torch.load(plan['teacher_checkpoint'], map_location='cpu', weights_only=True)
        teacher.load_state_dict(state['model'], strict=True)
        del state
        teacher.cuda().eval().requires_grad_(False)
        def capture_student(module, args, output):
            if module.training:
                student_states.append(output)
        def capture_teacher(module, args, output):
            value = output.detach().float()
            teacher_states.append(F.normalize(value - value.mean(-1, keepdim=True), dim=-1, eps=1e-8))
        for layer in model.network.bert.encoder.layer:
            hooks.append(layer.register_forward_hook(capture_student))
        for layer in teacher.bert.encoder.layer:
            hooks.append(layer.register_forward_hook(capture_teacher))
    opt, clipping = recovery.optimizer(model, trial)
    steps = math.ceil(len(train_data['labels']) / trial['batch_size']) * trial['epochs']
    scheduler = get_cosine_schedule_with_warmup(opt, math.ceil(steps * trial['warmup_ratio']), steps)
    few = {k: v[:4] for k, v in validation.items()}
    _, _, together = base.evaluate(model, few, search.JOB, 4, return_logits=True)
    _, _, separate = base.evaluate(model, few, search.JOB, 1, return_logits=True)
    torch.testing.assert_close(together, separate, atol=5e-4, rtol=5e-4)
    metrics, _, _ = base.evaluate(model, validation, search.JOB, trial['eval_batch_size'])
    best, stale = metrics['accuracy'], 0
    history = [{'epoch': 0, 'validation': metrics, 'branch_probe': recovery.branch_probe(model, validation)}]
    base.save_checkpoint(folder / 'best.pt', {'model': model.state_dict(), 'epoch': 0, 'validation': metrics, 'trial': trial})
    base.write_json(folder / 'initial_audit.json', {
        'projection_norms': search.audit(model), 'ordinary_layernorm': affine_audit(model),
        'replaced_layernorm_modules': replaced, 'batch_invariance_passed': True,
        'maximum_batch_logit_difference': (together-separate).abs().max().item(),
        'teacher_used_only_for_training': True, 'test_accessed': False})
    base.write_json(folder / 'run.json', {'trial': trial, 'pid': os.getpid(), 'gpu': torch.cuda.get_device_name(),
                                        'plan_sha256': base.file_sha(root / 'plan.json')})
    base.write_json(folder / 'history.json', history)
    print(json.dumps({'trial': trial['name'], **history[0]}), flush=True)
    base.seed_everything(42)
    for epoch in range(1, trial['epochs'] + 1):
        if history[-1]['epoch'] >= trial['min_epochs'] and stale >= trial['patience']:
            break
        model.train()
        started = time.monotonic()
        totals = {'objective': 0., 'ce': 0., 'kd': 0., 'representation': 0.}
        order = base.epoch_order(len(train_data['labels']), 42, 4000 + epoch)
        warmup = epoch <= trial['representation_warmup_epochs']
        for batch_number, (indices, batch) in enumerate(base.batches(train_data, trial['batch_size'], order)):
            opt.zero_grad(set_to_none=True)
            student_states.clear()
            teacher_states.clear()
            if teacher is not None:
                with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                    teacher(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask']).logits.float()
                ce = F.cross_entropy(logits, batch['labels'], label_smoothing=trial['smoothing'])
                temp = trial['temperature']
                kd = F.kl_div(F.log_softmax(logits / temp, dim=-1),
                              F.softmax(cache['logits'][indices].cuda() / temp, dim=-1), reduction='batchmean') * temp**2
                representation = logits.new_tensor(0.)
                if teacher is not None:
                    assert len(student_states) == len(teacher_states) == 12
                    terms = []
                    for student, target in zip(student_states, teacher_states):
                        assert student.shape == target.shape
                        value = student.float()
                        value = F.normalize(value - value.mean(-1, keepdim=True), dim=-1, eps=1e-8)
                        terms.append((1 - (value * target).sum(-1)).mean())
                    representation = torch.stack(terms).mean()
                loss = representation + 0.25 * kd if warmup else (
                    (1-trial['kd_weight'])*ce + trial['kd_weight']*kd + trial['representation_weight']*representation)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training objective')
            loss.backward()
            if epoch == 1 and batch_number == 0:
                checked = 0
                for module in model.modules():
                    if isinstance(module, nn.LayerNorm):
                        for param in (module.weight, module.bias):
                            assert param.grad is not None and torch.isfinite(param.grad).all()
                            assert param.grad.abs().max().item() > 0
                            checked += 1
                base.write_json(folder / 'layernorm_gradient_check.json', {'trainable_affine_tensors_with_nonzero_finite_gradients': checked})
                assert checked == 50
            for parameters in clipping.values():
                torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True, foreach=False)
            opt.step()
            recovery.retract(model)
            scheduler.step()
            for key, value in [('objective', loss), ('ce', ce), ('kd', kd), ('representation', representation)]:
                totals[key] += value.item() * len(indices)
            student_states.clear()
            teacher_states.clear()
        metrics, _, _ = base.evaluate(model, validation, search.JOB, trial['eval_batch_size'])
        norms = search.audit(model)
        if metrics['accuracy'] > best:
            best, stale = metrics['accuracy'], 0
            base.save_checkpoint(folder / 'best.pt', {'model': model.state_dict(), 'epoch': epoch,
                                 'validation': metrics, 'trial': trial})
        else:
            stale += 1
        record = {'epoch': epoch, 'validation': metrics,
                  'stage': 'representation_adaptation' if warmup else 'joint_training',
                  'train': {k: v / len(train_data['labels']) for k, v in totals.items()},
                  'maximum_effective_frobenius_fp64': max(x['frobenius_fp64'] for x in norms),
                  'seconds': time.monotonic() - started, 'best_validation_accuracy': best}
        history.append(record)
        base.write_json(folder / 'history.json', history)
        base.save_checkpoint(folder / 'latest.pt', {'model': model.state_dict(), 'optimizer': opt.state_dict(),
                             'scheduler': scheduler.state_dict(), 'rng': base.capture_rng(), 'history': history,
                             'trial': trial, 'plan_sha256': base.file_sha(root / 'plan.json')})
        print(json.dumps({'trial': trial['name'], **record}), flush=True)
    for hook in hooks:
        hook.remove()
    selected = torch.load(folder / 'best.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(selected['model'], strict=True)
    metrics, predictions, logits = base.evaluate(model, validation, search.JOB, trial['eval_batch_size'], return_logits=True)
    assert metrics == selected['validation']
    np.savez_compressed(folder / 'validation_predictions.npz', labels=validation['labels'].numpy(),
                        predictions=predictions.numpy(), logits=logits.numpy())
    base.write_json(folder / 'selected.json', {
        'trial': trial, 'selected_epoch': selected['epoch'], 'trained_epochs': history[-1]['epoch'],
        'validation': metrics, 'projection_norms': search.audit(model), 'ordinary_layernorm': affine_audit(model),
        'checkpoint_sha256': base.file_sha(folder / 'best.pt'),
        'validation_predictions_sha256': base.file_sha(folder / 'validation_predictions.npz'),
        'branch_probe': recovery.branch_probe(model, validation), 'test_evaluated': False})
    print(json.dumps({'training_complete': trial['name'], 'epoch': selected['epoch'], 'validation': metrics}), flush=True)


def evaluate_test(root, worker_id):
    base.configure()
    plan = verify(root)
    trial = next(t for t in plan['trials'] if t['worker'] == worker_id)
    folder = root / trial['name']
    selected = json.loads((folder / 'selected.json').read_text())
    history = json.loads((folder / 'history.json').read_text())
    best = max(history, key=lambda x: x['validation']['accuracy'])
    assert best['epoch'] == selected['selected_epoch'] and best['validation'] == selected['validation']
    assert base.file_sha(folder / 'best.pt') == selected['checkpoint_sha256']
    lock = {'selection': selected, 'plan_sha256': base.file_sha(root / 'plan.json'),
            'rule': plan['selection'], 'prior_test_disclosure': plan['prior_test_disclosure']}
    lock_path = folder / 'selection_lock.json'
    if lock_path.exists():
        assert json.loads(lock_path.read_text()) == lock
    else:
        base.write_json(lock_path, lock)
    model, _ = build(plan, trial)
    state = torch.load(folder / 'best.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(state['model'], strict=True)
    original = Path(plan['original_study'])
    for split in ('validation', 'test'):
        assert base.file_sha(original / 'data/genomic' / f'{split}.pt') == plan['data_sha256'][split]
    val = torch.load(original / 'data/genomic/validation.pt', weights_only=True)
    metrics, predictions, _ = base.evaluate(model, val, search.JOB, trial['eval_batch_size'])
    assert metrics == selected['validation']
    with np.load(folder / 'validation_predictions.npz') as saved:
        assert np.array_equal(predictions.numpy(), saved['predictions'])
    test = torch.load(original / 'data/genomic/test.pt', weights_only=True)
    metrics, predictions, logits = base.evaluate(model, test, search.JOB, trial['eval_batch_size'], return_logits=True)
    y, p = test['labels'].numpy(), predictions.numpy()
    assert len(y) == 9034 and torch.isfinite(logits).all()
    counts = {'tp': int(((y == 1) & (p == 1)).sum()), 'tn': int(((y == 0) & (p == 0)).sum()),
              'fp': int(((y == 0) & (p == 1)).sum()), 'fn': int(((y == 1) & (p == 0)).sum())}
    assert metrics['accuracy'] == (counts['tp'] + counts['tn']) / len(y)
    assert abs(metrics['f1'] - 2*counts['tp'] / (2*counts['tp'] + counts['fp'] + counts['fn'])) < 1e-12
    np.savez_compressed(folder / 'test_predictions.npz', labels=y, predictions=p, logits=logits.numpy())
    result = {**selected, 'test_evaluated': True, 'test': metrics, 'confusion_counts': counts,
              'projection_norms': search.audit(model), 'ordinary_layernorm': affine_audit(model),
              'selection_lock_sha256': base.file_sha(lock_path),
              'test_predictions_sha256': base.file_sha(folder / 'test_predictions.npz'),
              'prior_test_disclosure': plan['prior_test_disclosure']}
    base.write_json(folder / 'result.json', result)
    print(json.dumps({'test_complete': trial['name'], 'validation': selected['validation'], 'test': metrics}), flush=True)


def assemble(root):
    plan = verify(root)
    repository = Path(plan['repository'])
    reports = [json.loads((root / t['name'] / 'result.json').read_text()) for t in plan['trials']]
    chosen = max(reports, key=lambda r: r['validation']['accuracy'])
    out = repository / 'artifacts/rigorous_rerun/trainable_ln_fn1'
    out.mkdir(exist_ok=False)
    shutil.copy2(root / 'plan.json', out / 'plan.json')
    shutil.copy2(__file__, out / Path(__file__).name)
    for report in reports:
        src = root / report['trial']['name']
        dst = out / report['trial']['name']
        dst.mkdir()
        for name in ('result.json', 'selected.json', 'history.json', 'initial_audit.json',
                     'layernorm_gradient_check.json', 'selection_lock.json',
                     'validation_predictions.npz', 'test_predictions.npz'):
            shutil.copy2(src / name, dst / name)
    summary = {'validation_selected_variant': chosen['trial']['name'], 'seed': 42,
               'reports': reports, 'selection': plan['selection']}
    base.write_json(out / 'summary.json', summary)
    base.write_json(out / 'complete.json', {
        'summary': str(out / 'summary.json'),
        'validation_selected_variant': chosen['trial']['name'],
        'test': chosen['test'], 'test_variants': len(reports)})
    print(json.dumps({'complete': True, 'chosen': chosen['trial']['name'], 'test': chosen['test']}), flush=True)


def supervise(root):
    plan = verify(root)
    original = Path(plan['original_study'])
    processes = []
    env = {**os.environ, 'PYTHONPATH': f'{original}/source:{original}/source/src',
           'CUBLAS_WORKSPACE_CONFIG': ':4096:8', 'TOKENIZERS_PARALLELISM': 'false',
           'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'OMP_NUM_THREADS': '4',
           'MKL_NUM_THREADS': '4', 'TMPDIR': str(root / 'tmp')}
    (root / 'tmp').mkdir(exist_ok=True)
    for trial in plan['trials']:
        worker_env = {**env, 'CUDA_VISIBLE_DEVICES': str(trial['worker'])}
        log = (root / f"worker_{trial['worker']}.log").open('a')
        process = subprocess.Popen([sys.executable, '-u', str(root / Path(__file__).name),
                                    'run', '--root', str(root), '--worker', str(trial['worker'])],
                                   cwd=original / 'source', env=worker_env, stdout=log, stderr=subprocess.STDOUT)
        processes.append((process, log))
    base.write_json(root / 'processes.json', {'supervisor': os.getpid(), 'workers': [p.pid for p, _ in processes]})
    failed = []
    for process, log in processes:
        code = process.wait()
        log.close()
        if code:
            failed.append({'pid': process.pid, 'exit_code': code})
    if failed:
        base.write_json(root / 'failed.json', {'failed': failed})
        raise RuntimeError(f'Workers failed: {failed}')
    assemble(root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'supervise', 'train', 'test', 'run', 'assemble'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--worker', type=int, default=0)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.action == 'prepare':
        prepare(root)
    elif args.action == 'supervise':
        supervise(root)
    elif args.action == 'train':
        train(root, args.worker)
    elif args.action == 'test':
        evaluate_test(root, args.worker)
    elif args.action == 'run':
        train(root, args.worker)
        evaluate_test(root, args.worker)
    else:
        assemble(root)
