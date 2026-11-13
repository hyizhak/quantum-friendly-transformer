"""Validation-only FN1 recovery after finding an inactive gated FFN branch."""

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
spec = importlib.util.spec_from_file_location('frozen_fn1', PREVIOUS / 'unit_fn_search.py')
search = importlib.util.module_from_spec(spec)
spec.loader.exec_module(search)
base = search.base


class UnitGateActivation(nn.Module):
    """A stated architecture variant; normalize gated activations before W_out."""

    def forward(self, value):
        dtype = value.dtype
        with torch.autocast(device_type=value.device.type, enabled=False):
            value = F.normalize(value.float(), dim=-1, eps=1e-8)
        return value.to(dtype)


@torch.no_grad()
def retract(model):
    count = 0
    for module in model.modules():
        if hasattr(module, 'parametrizations') and 'weight' in module.parametrizations:
            raw = module.parametrizations.weight.original
            raw.div_(raw.norm().clamp_min(1e-12))
            count += 1
    assert count == 72


def build(plan, trial):
    old, original, study = search.load_plan(PREVIOUS)
    assert base.file_sha(Path(plan['checkpoint'])) == plan['checkpoint_sha256']
    model = search.build(old, original, study, trial).cuda()
    state = torch.load(plan['checkpoint'], map_location='cpu', weights_only=True)
    model.load_state_dict(state['model'], strict=True)
    del state
    before = [float(m.parametrizations.weight.original.norm()) for m in model.modules()
              if hasattr(m, 'parametrizations') and 'weight' in m.parametrizations]
    retract(model)
    if trial['midnorm']:
        for layer in model.network.bert.encoder.layer:
            layer.mlp.dropout = nn.Sequential(layer.mlp.dropout, UnitGateActivation())
    return model, before


def optimizer(model, trial):
    groups = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if 'parametrizations.weight.original' in name:
            kind = 'ffn' if '.mlp.' in name else 'attention'
            lr = trial['ffn_lr'] if kind == 'ffn' else trial['attention_lr']
            decay = 0.
        else:
            kind, lr = 'other', trial['other_lr']
            decay = 0.01 if parameter.ndim >= 2 else 0.
        groups.setdefault((kind, lr, decay), []).append(parameter)
    opt = torch.optim.AdamW([
        {'params': params, 'lr': lr, 'weight_decay': decay, 'kind': kind}
        for (kind, lr, decay), params in groups.items()
    ], eps=trial['adam_epsilon'], betas=(0.9, 0.999), foreach=False)
    clipping = {kind: [p for group in opt.param_groups if group['kind'] == kind for p in group['params']]
                for kind in ('ffn', 'attention', 'other')}
    return opt, clipping


@torch.inference_mode()
def branch_probe(model, validation):
    rows, hooks = [], []
    for index, layer in enumerate(model.network.bert.encoder.layer):
        def hook(module, args, output, index=index):
            signal = F.linear(args[0], module.weight, None)
            rows.append({'layer': index, 'linear_signal_norm': signal.norm(dim=-1).mean().item(),
                         'bias_norm': module.bias.norm().item()})
        hooks.append(layer.mlp.wo.register_forward_hook(hook))
    few = {key: value[:8] for key, value in validation.items()}
    base.evaluate(model, few, search.JOB, 8)
    for hook in hooks:
        hook.remove()
    return {'layers': rows, 'mean_linear_signal_norm': float(np.mean([x['linear_signal_norm'] for x in rows])),
            'mean_bias_norm': float(np.mean([x['bias_norm'] for x in rows]))}


def prepare(root):
    root.mkdir(exist_ok=False)
    old, original, study = search.load_plan(PREVIOUS)
    repository = Path(old['repository'])
    checkpoint = PREVIOUS / 'fn1_recovery_kd/best.pt'
    trial_base = {**old['trials'][0], 'epochs': 32, 'min_epochs': 12, 'patience': 8,
                  'warmup_ratio': 0.05, 'attention_lr': 1e-5, 'ffn_lr': 1e-4,
                  'other_lr': 3e-5, 'adam_epsilon': 1e-12}
    trial_base.pop('lr')
    trial_base.pop('aux_lr')
    trials = [{**trial_base, 'name': 'fn1_projected_adam', 'worker': 0, 'midnorm': False},
              {**trial_base, 'name': 'fn1_projected_adam_midnorm', 'worker': 1, 'midnorm': True}]
    snapshot = root / Path(__file__).name
    shutil.copy2(__file__, snapshot)
    plan = {
        'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'repository': str(repository), 'previous_plan_sha256': base.file_sha(PREVIOUS / 'plan.json'),
        'runner_sha256': base.file_sha(snapshot), 'checkpoint': str(checkpoint),
        'checkpoint_sha256': base.file_sha(checkpoint), 'original_study': str(original),
        'data_sha256': old['data_sha256'], 'trials': trials,
        'diagnostic_evidence_sha256': base.file_sha(repository / 'artifacts/rigorous_rerun/unit_fn_diagnostics.json'),
        'constraints': old['constraints'],
        'optimization': 'Rescale latent matrices to Frobenius norm one before training and after every AdamW step; this leaves the normalized forward map unchanged up to floating-point rounding. Separate gradient clipping for FFN matrices, attention matrices and other parameters; Adam epsilon 1e-12.',
        'midnorm_variant': 'Only the second variant adds L2 normalization of the post-dropout gated activation before the FFN output matrix. This changes the architecture and must be reported explicitly. It adds no trainable gain and keeps every complete projection Frobenius norm <=1. Its normalization conditioning is not a theorem-runtime guarantee.',
        'selection': 'Best validation accuracy within each variant, first tie wins, including epoch zero. No automatic test evaluations.',
        'prior_test_disclosure': 'The original experiment and four earlier FN1 follow-up test results have already been observed. These validation-only runs are subsequent exploratory optimization.',
    }
    base.write_json(root / 'plan.json', plan)
    base.write_json(repository / 'artifacts/rigorous_rerun/unit_fn_ffn_recovery_plan.json', plan)


def worker(root, worker_id):
    base.configure()
    plan = json.loads((root / 'plan.json').read_text())
    assert base.file_sha(__file__) == plan['runner_sha256']
    assert base.file_sha(PREVIOUS / 'plan.json') == plan['previous_plan_sha256']
    original = Path(plan['original_study'])
    for split in ('train', 'validation'):
        assert base.file_sha(original / 'data/genomic' / f'{split}.pt') == plan['data_sha256'][split]
    trial = next(t for t in plan['trials'] if t['worker'] == worker_id)
    folder = root / trial['name']
    folder.mkdir(exist_ok=True)
    if (folder / 'selected.json').exists():
        return
    base.seed_everything(42)
    model, raw_norms = build(plan, trial)
    train_data = torch.load(original / 'data/genomic/train.pt', weights_only=True)
    validation = torch.load(original / 'data/genomic/validation.pt', weights_only=True)
    teacher_record = json.loads((PREVIOUS / 'teacher_cache.json').read_text())
    assert base.file_sha(PREVIOUS / 'teacher_train.pt') == teacher_record['sha256']
    teacher = torch.load(PREVIOUS / 'teacher_train.pt', weights_only=True)
    opt, clipping = optimizer(model, trial)
    steps = math.ceil(len(train_data['labels']) / trial['batch_size']) * trial['epochs']
    scheduler = get_cosine_schedule_with_warmup(opt, math.ceil(steps * trial['warmup_ratio']), steps)
    norms = search.audit(model)
    few = {key: value[:4] for key, value in validation.items()}
    _, _, a = base.evaluate(model, few, search.JOB, 4, return_logits=True)
    _, _, b = base.evaluate(model, few, search.JOB, 1, return_logits=True)
    torch.testing.assert_close(a, b, atol=2e-4, rtol=2e-4)
    metrics, _, _ = base.evaluate(model, validation, search.JOB, trial['eval_batch_size'])
    if not trial['midnorm']:
        previous = json.loads((PREVIOUS / 'fn1_recovery_kd/selected.json').read_text())
        assert metrics['accuracy'] == previous['validation']['accuracy']
    history = [{'epoch': 0, 'validation': metrics, 'branch_probe': branch_probe(model, validation)}]
    best, stale = metrics['accuracy'], 0
    base.save_checkpoint(folder / 'best.pt', {'model': model.state_dict(), 'epoch': 0, 'validation': metrics, 'trial': trial})
    base.write_json(folder / 'initial_audit.json', {'norms': norms, 'batch_invariance_passed': True,
                    'raw_weight_norms_before_canonicalization': raw_norms, 'test_accessed': False})
    base.write_json(folder / 'history.json', history)
    base.write_json(folder / 'run.json', {'trial': trial, 'pid': os.getpid(), 'gpu': torch.cuda.get_device_name(),
                                        'plan_sha256': base.file_sha(root / 'plan.json')})
    print(json.dumps({'trial': trial['name'], **history[-1]}), flush=True)
    base.seed_everything(42)
    for epoch in range(1, trial['epochs'] + 1):
        if epoch > trial['min_epochs'] and stale >= trial['patience']:
            break
        model.train()
        started, total = time.monotonic(), 0.
        max_grad = {key: 0. for key in clipping}
        order = base.epoch_order(len(train_data['labels']), 42, 2000 + epoch)
        for indices, batch in base.batches(train_data, trial['batch_size'], order):
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask']).logits.float()
                ce = F.cross_entropy(logits, batch['labels'], label_smoothing=trial['smoothing'])
                temperature = trial['temperature']
                target = teacher['logits'][indices].cuda()
                kd = F.kl_div(F.log_softmax(logits / temperature, dim=-1),
                              F.softmax(target / temperature, dim=-1), reduction='batchmean') * temperature**2
                loss = (1 - trial['kd_weight']) * ce + trial['kd_weight'] * kd
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite loss')
            loss.backward()
            for kind, parameters in clipping.items():
                gradient = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True, foreach=False)
                max_grad[kind] = max(max_grad[kind], gradient.item())
            opt.step()
            retract(model)
            scheduler.step()
            total += loss.item() * len(indices)
        metrics, _, _ = base.evaluate(model, validation, search.JOB, trial['eval_batch_size'])
        norms = search.audit(model)
        if metrics['accuracy'] > best:
            best, stale = metrics['accuracy'], 0
            base.save_checkpoint(folder / 'best.pt', {'model': model.state_dict(), 'epoch': epoch,
                                 'validation': metrics, 'trial': trial})
        else:
            stale += 1
        record = {'epoch': epoch, 'validation': metrics, 'train_objective': total / len(train_data['labels']),
                  'max_gradient_norm_by_group': max_grad,
                  'maximum_effective_frobenius_fp64': max(x['frobenius_fp64'] for x in norms),
                  'branch_probe': branch_probe(model, validation), 'seconds': time.monotonic() - started,
                  'best_validation_accuracy': best}
        history.append(record)
        base.write_json(folder / 'history.json', history)
        base.save_checkpoint(folder / 'latest.pt', {'model': model.state_dict(), 'optimizer': opt.state_dict(),
                             'scheduler': scheduler.state_dict(), 'rng': base.capture_rng(), 'history': history,
                             'trial': trial, 'plan_sha256': base.file_sha(root / 'plan.json')})
        print(json.dumps({'trial': trial['name'], **record}), flush=True)
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
