"""Fresh teacher initialization and token-wise layer distillation under FN1."""

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
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
    root.mkdir(exist_ok=False)
    old, original, study = search.load_plan(PREVIOUS)
    repository = Path(old['repository'])
    common = {**old['trials'][0], 'initialization': 'teacher12', 'epochs': 24,
              'min_epochs': 12, 'patience': 8, 'warmup_ratio': 0.05,
              'representation_warmup_epochs': 2, 'representation_weight': 0.25,
              'attention_lr': 1e-5, 'ffn_lr': 1e-4, 'other_lr': 2e-4,
              'adam_epsilon': 1e-12, 'midnorm': True}
    common.pop('lr')
    common.pop('aux_lr')
    trials = [{**common, 'name': 'fn1_layerwise_preserve_bias', 'worker': 0, 'calibrate_bias': False},
              {**common, 'name': 'fn1_layerwise_calibrate_bias', 'worker': 1, 'calibrate_bias': True}]
    snapshot = root / Path(__file__).name
    shutil.copy2(__file__, snapshot)
    plan = {
        'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'runner_sha256': base.file_sha(snapshot), 'recovery_dependency_sha256': base.file_sha(RECOVERY),
        'previous_plan_sha256': base.file_sha(PREVIOUS / 'plan.json'),
        'repository': str(repository), 'original_study': str(original),
        'teacher_checkpoint': old['teacher_checkpoint'], 'teacher_sha256': old['teacher_sha256'],
        'data_sha256': old['data_sha256'], 'constraints': old['constraints'], 'trials': trials,
        'initialization': 'Fresh conversion of the seed42 native DNABERT-2 teacher selected at epoch 12; not continuation of the overfit FN model.',
        'representation_loss': 'Mean cosine distance between centered unit-normalized teacher token states and student token states, averaged over all 12 layers and all non-padding training tokens. Teacher runs frozen in evaluation mode. The first two epochs use representation loss plus 0.25 times logit KD; later epochs use 0.5 CE + 0.5 KD + 0.25 representation loss.',
        'bias_calibration': 'In the second variant only, initialize each normalized linear projection bias divided by its corresponding original raw weight Frobenius norm. Subsequent biases remain trainable, as in the original normalization scope.',
        'architecture_change': 'Both variants add L2 normalization of the post-dropout gated activation before the FFN output matrix. All complete projection matrices remain bounded by one. This extra nonlinear normalization and its conditioning must be disclosed; no theorem-runtime guarantee is inferred.',
        'data_scope': 'Only existing training sequences are used for representation adaptation and supervised training. No external pretraining corpus has been downloaded or used in this run.',
        'selection': 'Highest validation accuracy, first tie wins, including epoch zero; no automatic test evaluation.',
        'prior_test_disclosure': 'The original and four previous FN1 follow-up test results have already been observed. These are subsequent exploratory validation-only experiments.',
        'method_reference': 'https://arxiv.org/abs/1909.10351 (motivation for intermediate-layer distillation; this experiment is not a reproduction of TinyBERT)',
    }
    base.write_json(root / 'plan.json', plan)
    base.write_json(repository / 'artifacts/rigorous_rerun/unit_fn_layerwise_plan.json', plan)


def build_student(plan, trial):
    old, original, study = search.load_plan(PREVIOUS)
    model = search.build(old, original, study, trial).cuda()
    calibration = []
    if trial['calibrate_bias']:
        with torch.no_grad():
            for name, module in model.named_modules():
                if hasattr(module, 'parametrizations') and 'weight' in module.parametrizations and module.bias is not None:
                    norm = module.parametrizations.weight.original.norm()
                    before = module.bias.norm().item()
                    module.bias.div_(norm.clamp_min(1e-12))
                    calibration.append({'module': name, 'weight_norm': norm.item(),
                                        'bias_norm_before': before, 'bias_norm_after': module.bias.norm().item()})
    recovery.retract(model)
    for layer in model.network.bert.encoder.layer:
        layer.mlp.dropout = nn.Sequential(layer.mlp.dropout, recovery.UnitGateActivation())
    return model, calibration


def worker(root, worker_id):
    base.configure()
    plan = json.loads((root / 'plan.json').read_text())
    assert base.file_sha(__file__) == plan['runner_sha256']
    assert base.file_sha(RECOVERY) == plan['recovery_dependency_sha256']
    assert base.file_sha(PREVIOUS / 'plan.json') == plan['previous_plan_sha256']
    old, original, study = search.load_plan(PREVIOUS)
    for split in ('train', 'validation'):
        assert base.file_sha(original / 'data/genomic' / f'{split}.pt') == plan['data_sha256'][split]
    assert base.file_sha(Path(plan['teacher_checkpoint'])) == plan['teacher_sha256']
    trial = next(t for t in plan['trials'] if t['worker'] == worker_id)
    folder = root / trial['name']
    folder.mkdir(exist_ok=True)
    if (folder / 'selected.json').exists():
        return
    if (folder / 'latest.pt').exists():
        raise RuntimeError('Existing partial run must be explicitly resumed rather than overwritten')
    base.seed_everything(42)
    model, calibration = build_student(plan, trial)
    train_data = torch.load(original / 'data/genomic/train.pt', weights_only=True)
    validation = torch.load(original / 'data/genomic/validation.pt', weights_only=True)
    teacher, _ = base.build_model(study, original, {**search.JOB, 'variant': 'vanilla'})
    state = torch.load(plan['teacher_checkpoint'], map_location='cpu', weights_only=True)
    teacher.load_state_dict(state['model'], strict=True)
    del state
    teacher.cuda().eval().requires_grad_(False)
    cached_record = json.loads((PREVIOUS / 'teacher_cache.json').read_text())
    assert base.file_sha(PREVIOUS / 'teacher_train.pt') == cached_record['sha256']
    cache = torch.load(PREVIOUS / 'teacher_train.pt', weights_only=True)
    student_states, teacher_states, hooks = [], [], []
    def capture_student(module, args, output):
        if module.training:
            student_states.append(output)
    def capture_teacher(module, args, output):
        value = output.detach().float()
        value = F.normalize(value - value.mean(-1, keepdim=True), dim=-1, eps=1e-8)
        teacher_states.append(value)
    for layer in model.network.bert.encoder.layer:
        hooks.append(layer.register_forward_hook(capture_student))
    for layer in teacher.bert.encoder.layer:
        hooks.append(layer.register_forward_hook(capture_teacher))
    opt, clipping = recovery.optimizer(model, trial)
    total_steps = math.ceil(len(train_data['labels']) / trial['batch_size']) * trial['epochs']
    scheduler = get_cosine_schedule_with_warmup(opt, math.ceil(total_steps * trial['warmup_ratio']), total_steps)
    norms = search.audit(model)
    few = {key: value[:4] for key, value in validation.items()}
    _, _, a = base.evaluate(model, few, search.JOB, 4, return_logits=True)
    _, _, b = base.evaluate(model, few, search.JOB, 1, return_logits=True)
    torch.testing.assert_close(a, b, atol=2e-4, rtol=2e-4)
    metrics, _, _ = base.evaluate(model, validation, search.JOB, trial['eval_batch_size'])
    best, stale = metrics['accuracy'], 0
    history = [{'epoch': 0, 'validation': metrics, 'branch_probe': recovery.branch_probe(model, validation)}]
    base.save_checkpoint(folder / 'best.pt', {'model': model.state_dict(), 'epoch': 0, 'validation': metrics, 'trial': trial})
    base.write_json(folder / 'initial_audit.json', {'norms': norms, 'batch_invariance_passed': True,
                    'bias_calibration': calibration, 'teacher_frozen': True, 'test_accessed': False})
    base.write_json(folder / 'history.json', history)
    base.write_json(folder / 'run.json', {'trial': trial, 'pid': os.getpid(), 'gpu': torch.cuda.get_device_name(),
                                        'plan_sha256': base.file_sha(root / 'plan.json')})
    print(json.dumps({'trial': trial['name'], 'epoch': 0, 'validation': metrics}), flush=True)
    base.seed_everything(42)
    for epoch in range(1, trial['epochs'] + 1):
        if epoch > trial['min_epochs'] and stale >= trial['patience']:
            break
        model.train()
        started = time.monotonic()
        totals = {'objective': 0., 'representation_loss': 0., 'ce': 0., 'kd': 0.}
        order = base.epoch_order(len(train_data['labels']), 42, 3000 + epoch)
        warmup = epoch <= trial['representation_warmup_epochs']
        for indices, batch in base.batches(train_data, trial['batch_size'], order):
            opt.zero_grad(set_to_none=True)
            student_states.clear()
            teacher_states.clear()
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                teacher(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask']).logits.float()
                assert len(student_states) == len(teacher_states) == 12
                terms = []
                for student, target in zip(student_states, teacher_states):
                    assert student.shape == target.shape and student.shape[0] == int(batch['attention_mask'].sum())
                    terms.append((1 - (F.normalize(student.float(), dim=-1) * target).sum(-1)).mean())
                representation = torch.stack(terms).mean()
                ce = F.cross_entropy(logits, batch['labels'], label_smoothing=trial['smoothing'])
                temperature = trial['temperature']
                target_logits = cache['logits'][indices].cuda()
                kd = F.kl_div(F.log_softmax(logits / temperature, dim=-1),
                              F.softmax(target_logits / temperature, dim=-1), reduction='batchmean') * temperature**2
                loss = representation + 0.25 * kd if warmup else 0.5 * ce + 0.5 * kd + trial['representation_weight'] * representation
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite layerwise distillation loss')
            loss.backward()
            for parameters in clipping.values():
                torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True, foreach=False)
            opt.step()
            recovery.retract(model)
            scheduler.step()
            for key, value in [('objective', loss), ('representation_loss', representation), ('ce', ce), ('kd', kd)]:
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
        record = {'epoch': epoch, 'validation': metrics, 'stage': 'representation_adaptation' if warmup else 'joint_training',
                  'train': {key: value / len(train_data['labels']) for key, value in totals.items()},
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
    base.write_json(folder / 'selected.json', {'trial': trial, 'selected_epoch': selected['epoch'],
                    'trained_epochs': history[-1]['epoch'], 'validation': metrics, 'norms': search.audit(model),
                    'checkpoint_sha256': base.file_sha(folder / 'best.pt'),
                    'validation_predictions_sha256': base.file_sha(folder / 'validation_predictions.npz'),
                    'test_evaluated': False})
    print(json.dumps({'complete': trial['name'], 'selected_epoch': selected['epoch'], 'validation': metrics}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'worker'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--worker', type=int, default=0)
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(args.root.resolve())
    else:
        worker(args.root.resolve(), args.worker)
