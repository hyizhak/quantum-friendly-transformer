"""Bounded continuation with training-only distillation and trajectory averaging."""

import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

import torch

from experiments.lib import recovery_guided as guided
from experiments.lib import recovery_full_dense as dense
from experiments.lib import recovery_root_data as roots
from experiments.lib.recovery_extension import validate_scale_records
from experiments.lib.reviewer import manifest, save_csv, save_json
from experiments.runners import reviewer_recovery_extension as core
from experiments.runners import reviewer_recovery_large as large
from experiments.runners import reviewer_recovery_full_dense as full_dense
from experiments.runners import reviewer_recovery_matched as matched
from quantum_friendly_transformer.norm_transformer import (
    collect_attention_scaling_stats, collect_learnable_cap_stats, project_attention_scales_,
)
from quantum_friendly_transformer.norm_transformer.frobenius_cap import target_weight_modules


def load_parent(model, config, name):
    reference = config['parent_checkpoint']
    assert dense.file_sha256(reference['path']) == reference['sha256']
    checkpoint = torch.load(reference['path'], map_location='cpu', weights_only=True, mmap=True)
    assert checkpoint['model'] == name and checkpoint['mode'] == 'bounded'
    assert checkpoint['config']['cap'] == config['cap']
    for key in ['model_id', 'revision', 'alpha_s', 'baseline_target']:
        assert checkpoint['config']['models'][name][key] == config['models'][name][key], key
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    assert all(p.requires_grad for p in model.parameters())


def build_model(assets, config, name, seed, devices):
    if name == 'mistral7b':
        model, caps, adapters, _, _ = full_dense.build_model(assets, config, seed, devices, initialize_parent=False)
    elif name == 'olmo1b':
        model, caps, adapters = large.build_model(assets, name, config, seed, 'bounded', devices[0])
    else:
        model, caps, adapters = core.build_model(assets, name, config, seed, 'bounded', devices[0])
    load_parent(model, config, name)
    return model, caps, adapters


def make_optimizers(model, config, name):
    groups, scales = guided.parameter_groups(model, config['models'][name], config['weight_decay'])
    if name == 'mistral7b':
        import bitsandbytes as bnb
        assert bnb.__version__ == config['optimizer_version']
        optimizer = bnb.optim.AdamW8bit(groups)
    else:
        optimizer = torch.optim.AdamW(groups, fused=True)
    scale_optimizer = torch.optim.AdamW(scales, lr=config['scale_learning_rate'],
                                       weight_decay=0., eps=1e-12, foreach=False)
    return optimizer, scale_optimizer, [p for group in groups for p in group['params']], scales


def main(args):
    config = json.loads(args.config.read_text())
    recipe = config['models'][args.model]
    if args.smoke:
        recipe['steps'] = 2
        config.update(eval_every=2, log_every=1, average_start=1, average_every=1)
    args.output.mkdir(parents=True, exist_ok=False)
    save_json(args.output / 'config.json', config)
    torch.set_num_threads(config['threads'])
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    devices = [torch.device(d) for d in args.devices]
    device = devices[0]
    started = time.monotonic()
    tokenizer, data, split = matched.prepare_data(args.assets, args.model, config)
    root_ids = set(json.loads(roots.ALIGNMENT.read_text())['glue_training_example_ids'])
    root_indices = torch.tensor([i for i, value in enumerate(data['train']['example_id'].tolist()) if value in root_ids])
    assert len(root_indices) == 6803
    split.update(training_sampling_protocol=config['training_sampling_protocol'],
                 root_sampling_fraction=config['root_fraction'], root_pool_rows=len(root_indices))
    teacher_metadata = None
    targets = None
    if config['distillation_weight'] > 0:
        teacher_metadata = json.loads((args.teacher / 'manifest.json').read_text())
        asset_manifest = json.loads((args.assets / 'assets.json').read_text())
        assert teacher_metadata['dataset_revision'] == config['dataset_revision']
        assert teacher_metadata['training_data_sha256'] == asset_manifest['sha256']['data/sst2/train-00000-of-00001.parquet']
        assert dense.file_sha256(args.teacher / 'training_logits.pth') == teacher_metadata['training_logits_sha256']
        cache = torch.load(args.teacher / 'training_logits.pth', weights_only=True, map_location='cpu')
        targets = guided.align_training_targets(data['train'], cache).to(device)
        save_json(args.output / 'teacher_reference.json', {
            'manifest_path': str(args.teacher / 'manifest.json'),
            'manifest_sha256': dense.file_sha256(args.teacher / 'manifest.json'),
            'training_logits_sha256': teacher_metadata['training_logits_sha256'],
            'teacher_id': teacher_metadata['teacher_id'], 'teacher_revision': teacher_metadata['teacher_revision'],
            'student_training_rows': len(targets), 'validation_targets_used_for_training': False})
    if args.smoke:
        for split_name in ['selection', 'evaluation']:
            data[split_name] = {k: v[:32] for k, v in data[split_name].items()}
    save_json(args.output / 'split.json', split)
    data = {name: {k: v.to(device) for k, v in batch.items()} for name, batch in data.items()}
    model, caps, adapters = build_model(args.assets, config, args.model, args.seed, devices)
    model.config.pad_token_id = tokenizer.pad_token_id
    save_json(args.output / 'parameter_inventory.json', dense.parameter_inventory(model))
    if args.model in ['olmo1b', 'mistral7b']:
        core.AttentionInputMonitor = large.LargeAttentionInputMonitor
    alpha = recipe['alpha_s']
    sources = [Path(__file__), Path(guided.__file__), Path(dense.__file__), Path(core.__file__),
               Path(large.__file__), Path(full_dense.__file__), Path(matched.__file__), roots.ALIGNMENT,
               ROOT / 'experiments/lib/reviewer.py', ROOT / 'experiments/lib/recovery_extension.py',
               ROOT / 'src/quantum_friendly_transformer/norm_transformer/frobenius_cap.py',
               ROOT / 'src/quantum_friendly_transformer/norm_transformer/attention_scaling.py',
               ROOT / 'src/quantum_friendly_transformer/experiments/runners/recovery/glue_sequence_classifier.py']
    metadata = manifest(config, sources, model=args.model, seed=args.seed, mode='bounded',
        assets=json.loads((args.assets / 'assets.json').read_text()), devices=[str(d) for d in devices],
        train_scope='all dense model parameters', smoke=args.smoke,
        selection_protocol='official_validation', training_sampling_protocol=config['training_sampling_protocol'],
        initialization='selected bounded parent; fresh optimizer; training-trajectory parameter averages',
        parent_checkpoint=config['parent_checkpoint'], teacher=teacher_metadata,
        seed_role='continued-training order; fixed selected parent',
        limitations=['Validation selects checkpoints and live/averaged variants; these are exploratory examples, not independent test scores.',
                     'Caps and historical alpha floors remain fixed; the input alpha constants are not certified full-matrix bounds.',
                     'Distillation uses an external SST-2-finetuned teacher when enabled; this changes the training resources.',
                     'Parameter averaging is along this continuation only; inference uses a single unchanged student model.'])
    save_json(args.output / 'running_manifest.json', metadata)
    evaluations, training, scale_rows, weight_rows, norm_rows = [], [], [], [], []
    best_key, best_state, best_scores, best_step, best_variant = (-math.inf, -math.inf), None, None, None, None
    average = guided.ParameterAverage(model)

    def evaluate_variant(step, variant):
        nonlocal best_key, best_state, best_scores, best_step, best_variant
        scores, predictions, losses, norms = core.evaluate(model, data['selection'], config['eval_batch_size'],
                                                          alpha, device, monitor=True)
        scales = collect_attention_scaling_stats(model)
        validate_scale_records(scales, alpha, config['cap'])
        with torch.no_grad():
            maximum = max(float(module.weight.float().norm()) for _, module in target_weight_modules(model))
        assert maximum <= config['cap'] * (1 + 1e-6)
        evaluations.append({'step': step, 'variant': variant, 'average_count': average.count if variant == 'average' else 0,
                            'seconds': time.monotonic()-started, 'maximum_effective_weight_norm': maximum, **scores})
        scale_rows.extend({'step': step, 'variant': variant, **r} for r in scales)
        weight_rows.extend({'step': step, 'variant': variant, **r} for r in collect_learnable_cap_stats(model))
        norm_rows.extend({'step': step, 'variant': variant, **r} for r in norms)
        key = (scores['accuracy'], -scores['mean_nll'])
        if key > best_key:
            best_key, best_scores, best_step, best_variant = key, scores, step, variant
            # Retain the best in host RAM and write it once at completion. This
            # preserves previous runs and avoids two additional 27GiB files.
            best_state = guided.clone_cpu_state(model)
            save_csv(args.output / 'predictions.csv', [
                {'example_id': int(i), 'label': int(y), 'prediction': int(p), 'nll': float(nll)}
                for i, y, p, nll in zip(data['selection']['example_id'].cpu(), data['selection']['labels'].cpu(), predictions, losses)])
            save_json(args.output / 'selected_scale_stats.json', scales)
            save_csv(args.output / 'evaluation_attention_inputs.csv', norms)
        for name, rows in [('selection', evaluations), ('scales', scale_rows), ('weights', weight_rows), ('attention_inputs', norm_rows)]:
            save_csv(args.output / f'{name}.csv', rows)
        print(f'{args.model} step={step} variant={variant} score={scores["correct"]}/{scores["examples"]} '
              f'best={best_scores["correct"]} elapsed={time.monotonic()-started:.1f}s', flush=True)
        model.train()
        return scores['accuracy'] >= recipe['baseline_target'] - .01

    evaluate_variant(0, 'live')
    optimizer, scale_optimizer, ordinary, scales = make_optimizers(model, config, args.model)
    train_model = torch.compile(model, mode='reduce-overhead', dynamic=False) if config.get('compile') and not args.smoke else model
    generator = torch.Generator().manual_seed(args.seed + 8917)
    size = len(data['train']['labels'])
    warmup = max(1, int(recipe['steps'] * config['warmup_fraction']))
    micro = config['micro_batch_size']
    accumulation = config['gradient_accumulation_steps']
    assert micro * accumulation == config['batch_size']
    before = dense.parameter_probes(model)
    completed_steps = 0
    recovered = False
    for step in range(1, recipe['steps'] + 1):
        tick = time.monotonic()
        multiplier = guided.cyclic_multiplier(step, warmup, config['cycle_steps'], config['minimum_lr_fraction'])
        for group in optimizer.param_groups:
            group['lr'] = recipe['learning_rate'] * group['lr_multiplier'] * multiplier
        scale_optimizer.param_groups[0]['lr'] = config['scale_learning_rate'] * multiplier
        optimizer.zero_grad(set_to_none=True)
        scale_optimizer.zero_grad(set_to_none=True)
        total_loss, total_ce, total_kd, correct = 0., 0., 0., 0
        for _ in range(accumulation):
            nroot = int(micro * config['root_fraction'])
            selected_roots = root_indices[torch.randint(len(root_indices), (nroot,), generator=generator)]
            indices = torch.cat([selected_roots, torch.randint(size, (micro - nroot,), generator=generator)]).to(device)
            batch = core.batch_at(data['train'], indices)
            if config.get('compile') and not args.smoke:
                torch.compiler.cudagraph_mark_step_begin()
            with torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
                output = train_model(**batch)
                loss, ce, kd = guided.guided_loss(output.logits, batch['labels'],
                    targets[indices] if targets is not None else None, config['distillation_weight'], config['temperature'])
            (loss / accumulation).backward()
            total_loss += float(loss.detach()) / accumulation
            total_ce += float(ce) / accumulation
            total_kd += float(kd) / accumulation
            correct += int((output.logits.detach().argmax(-1) == batch['labels']).sum())
            del output, loss
        norm = float(torch.nn.utils.clip_grad_norm_(ordinary, 1., error_if_nonfinite=True))
        torch.nn.utils.clip_grad_norm_(scales, 1., error_if_nonfinite=True)
        optimizer.step()
        scale_optimizer.step()
        project_attention_scales_(model, scale_optimizer)
        if step == 1:
            save_json(args.output / 'first_step_gradient_audit.json', dense.gradient_audit(model, before))
            del before
        completed_steps = step
        if step >= config['average_start'] and (step - config['average_start']) % config['average_every'] == 0:
            average.update(model)
        row = {'step': step, 'mean_training_loss': total_loss, 'gold_ce': total_ce, 'eligible_teacher_kl': total_kd,
               'training_accuracy': correct / config['batch_size'], 'top_learning_rate': recipe['learning_rate'] * multiplier,
               'gradient_norm_before_clip': norm, 'step_seconds': time.monotonic()-tick, 'seconds': time.monotonic()-started}
        training.append(row)
        if step % config['log_every'] == 0 or step == 1:
            save_csv(args.output / 'training.csv', training)
            save_json(args.output / 'progress.json', {'status': 'training', **row, 'maximum_steps': recipe['steps'],
                      'best_correct': best_scores['correct'], 'selected_step': best_step, 'selected_variant': best_variant})
            print(f'{args.model} train_step={step} loss={total_loss:.4f} seconds_per_step={row["step_seconds"]:.2f}', flush=True)
        if step % config['eval_every'] == 0 or step == recipe['steps']:
            optimizer.zero_grad(set_to_none=True)
            scale_optimizer.zero_grad(set_to_none=True)
            recovered = evaluate_variant(step, 'live')
            if average.count and (not recovered or args.smoke):
                live = guided.clone_cpu_state(model)
                try:
                    model.load_state_dict(average.state, strict=True)
                    recovered = evaluate_variant(step, 'average')
                finally:
                    model.load_state_dict(live, strict=True)
                    del live
            if recovered and not args.smoke:
                break
    save_csv(args.output / 'training.csv', training)
    checkpoint = {'model_state_dict': best_state, 'step': best_step, 'selected_variant': best_variant,
                  'selection_metrics': best_scores, 'model': args.model, 'mode': 'bounded', 'seed': args.seed, 'config': config}
    checkpoint_path = args.output / 'selected.pth'
    if not args.smoke:
        temporary = args.output / 'selected.tmp.pth'
        torch.save(checkpoint, temporary)
        os.replace(temporary, checkpoint_path)
    result = {'model': recipe['model_id'], 'name': args.model, 'mode': 'bounded', 'seed': args.seed,
              'selected_step': best_step, 'selected_variant': best_variant, 'selection_accuracy': best_key[0],
              'evaluation': best_scores, 'historical_baseline_target': recipe['baseline_target'],
              'historical_recovery_threshold': recipe['baseline_target'] - .01,
              'meets_historical_threshold': best_scores['accuracy'] >= recipe['baseline_target'] - .01,
              'alpha_s': alpha, 'alpha_w': config['cap'], 'configured_denominator_floor': alpha**2 * config['cap']**2,
              'maximum_observed_matrix_to_alpha_ratio': max(r['maximum_matrix_to_alpha_ratio'] for r in norm_rows),
              'applied_caps': caps, 'adapters': adapters, 'seconds': time.monotonic()-started,
              'smoke': args.smoke, 'checkpoint_sha256': dense.file_sha256(checkpoint_path) if not args.smoke else None,
              'selection_protocol': 'official_validation', 'training_sampling_protocol': config['training_sampling_protocol'],
              'initialization': metadata['initialization'], 'parent_checkpoint': config['parent_checkpoint'],
              'seed_role': metadata['seed_role'], 'train_scope': 'all dense model parameters',
              'trainable_parameters': sum(p.numel() for p in model.parameters()), 'optimizer_steps': completed_steps,
              'training_examples_seen': completed_steps * config['batch_size'], 'trajectory_snapshots': average.count,
              'external_teacher_used': teacher_metadata is not None}
    save_json(args.output / 'result.json', result)
    save_json(args.output / 'manifest.json', metadata)
    save_json(args.output / 'progress.json', {'status': 'complete', 'optimizer_steps': completed_steps,
              'selected_step': best_step, 'selected_variant': best_variant, 'evaluation': best_scores})
    print(json.dumps({k: result[k] for k in ['name', 'evaluation', 'selected_step', 'selected_variant', 'meets_historical_threshold']}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--teacher', type=Path, default=ROOT / 'results/reviewer/recovery_teacher')
    parser.add_argument('--model', choices=['2layers33m', 'olmo1b', 'mistral7b'], required=True)
    parser.add_argument('--seed', type=int, default=611)
    parser.add_argument('--devices', nargs='+', default=['cuda:0'])
    parser.add_argument('--smoke', action='store_true')
    main(parser.parse_args())
