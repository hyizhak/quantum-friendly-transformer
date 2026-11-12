"""Sharpness-aware full-parameter continuation of the missing two-layer model."""

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

import torch
from torch.nn import functional as F

from experiments.lib import recovery_sam as sam
from experiments.lib.recovery_extension import validate_scale_records
from experiments.lib.reviewer import manifest, save_csv, save_json
from experiments.runners import reviewer_recovery_guided as base
from quantum_friendly_transformer.norm_transformer import (
    collect_attention_scaling_stats, collect_learnable_cap_stats, project_attention_scales_,
)
from quantum_friendly_transformer.norm_transformer.frobenius_cap import target_weight_modules


def main(args):
    config = json.loads(args.config.read_text())
    name = '2layers33m'
    recipe = config['models'][name]
    if args.smoke:
        recipe['steps'] = 2
        config.update(eval_every=2, log_every=1, compile=False)
    args.output.mkdir(parents=True, exist_ok=False)
    save_json(args.output / 'config.json', config)
    torch.set_num_threads(config['threads'])
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device(args.device)
    started = time.monotonic()
    tokenizer, data, split = base.matched.prepare_data(args.assets, name, config)
    root_ids = set(json.loads(base.roots.ALIGNMENT.read_text())['glue_training_example_ids'])
    root_indices = torch.tensor([i for i, value in enumerate(data['train']['example_id'].tolist()) if value in root_ids])
    assert len(root_indices) == 6803
    split.update(training_sampling_protocol=config['training_sampling_protocol'],
                 root_sampling_fraction=config['root_fraction'], root_pool_rows=len(root_indices))
    if args.smoke:
        for part in ['selection', 'evaluation']:
            data[part] = {k: v[:32] for k, v in data[part].items()}
    save_json(args.output / 'split.json', split)
    data = {part: {k: v.to(device) for k, v in values.items()} for part, values in data.items()}
    model, caps, adapters = base.build_model(args.assets, config, name, args.seed, [device])
    model.config.pad_token_id = tokenizer.pad_token_id
    save_json(args.output / 'parameter_inventory.json', base.dense.parameter_inventory(model))
    sources = [Path(__file__), Path(sam.__file__), Path(base.__file__),
               *[Path(module.__file__) for module in [base.guided, base.dense, base.core, base.large,
                                                     base.full_dense, base.matched, base.roots]],
               base.roots.ALIGNMENT, ROOT / 'experiments/lib/reviewer.py',
               ROOT / 'experiments/lib/recovery_extension.py',
               ROOT / 'src/quantum_friendly_transformer/norm_transformer/frobenius_cap.py',
               ROOT / 'src/quantum_friendly_transformer/norm_transformer/attention_scaling.py',
               ROOT / 'src/quantum_friendly_transformer/experiments/runners/recovery/glue_sequence_classifier.py']
    metadata = manifest(config, sources, model=name, seed=args.seed, mode='bounded', smoke=args.smoke,
        assets=json.loads((args.assets / 'assets.json').read_text()),
        train_scope='all dense model parameters', selection_protocol='official_validation',
        training_sampling_protocol=config['training_sampling_protocol'],
        parent_checkpoint=config['parent_checkpoint'], external_teacher_used=False,
        initialization='selected bounded parent; fresh SAM/AdamW optimizer',
        seed_role='continued-training order; fixed selected parent',
        limitations=['Exploratory validation selection; not an independent test or seed average.',
                     'Fixed scalar floors do not certify the historical input-matrix bound.',
                     'The same gold-label minibatch and dropout randomness are used in both SAM passes.',
                     'This recipe jointly changes optimizer and sampling; no isolated causal effect is claimed.'])
    save_json(args.output / 'running_manifest.json', metadata)
    evaluations, scale_rows, weight_rows, norm_rows, training = [], [], [], [], []
    best_key, best_state, best_scores, best_step = (-math.inf, -math.inf), None, None, None
    alpha = recipe['alpha_s']

    def evaluate(step):
        nonlocal best_key, best_state, best_scores, best_step
        scores, predictions, losses, norms = base.core.evaluate(
            model, data['selection'], config['eval_batch_size'], alpha, device, monitor=True)
        scales = collect_attention_scaling_stats(model)
        validate_scale_records(scales, alpha, config['cap'])
        with torch.no_grad():
            maximum = max(float(module.weight.float().norm()) for _, module in target_weight_modules(model))
        assert maximum <= config['cap'] * (1 + 1e-6)
        evaluations.append({'step': step, 'variant': 'live', 'seconds': time.monotonic()-started,
                            'maximum_effective_weight_norm': maximum, **scores})
        scale_rows.extend({'step': step, **row} for row in scales)
        weight_rows.extend({'step': step, **row} for row in collect_learnable_cap_stats(model))
        norm_rows.extend({'step': step, **row} for row in norms)
        key = (scores['accuracy'], -scores['mean_nll'])
        if key > best_key:
            best_key, best_state, best_scores, best_step = key, base.guided.clone_cpu_state(model), scores, step
            save_csv(args.output / 'predictions.csv', [
                {'example_id': int(i), 'label': int(y), 'prediction': int(p), 'nll': float(nll)}
                for i, y, p, nll in zip(data['selection']['example_id'].cpu(),
                    data['selection']['labels'].cpu(), predictions, losses)])
            save_json(args.output / 'selected_scale_stats.json', scales)
            save_csv(args.output / 'evaluation_attention_inputs.csv', norms)
        for filename, rows in [('selection', evaluations), ('scales', scale_rows),
                               ('weights', weight_rows), ('attention_inputs', norm_rows)]:
            save_csv(args.output / f'{filename}.csv', rows)
        print(f'seed={args.seed} step={step} score={scores["correct"]}/{scores["examples"]} '
              f'best={best_scores["correct"]}', flush=True)
        model.train()
        return scores['accuracy'] >= recipe['baseline_target'] - .01

    evaluate(0)
    optimizer, scale_optimizer, ordinary, scales = base.make_optimizers(model, config, name)
    trained = torch.compile(model, mode='reduce-overhead', dynamic=False) if config['compile'] else model
    generator = torch.Generator().manual_seed(args.seed + 8917)
    size = len(data['train']['labels'])
    warmup = max(1, int(recipe['steps'] * config['warmup_fraction']))
    before = base.dense.parameter_probes(model)
    completed = 0
    for step in range(1, recipe['steps'] + 1):
        tick = time.monotonic()
        multiplier = base.guided.cyclic_multiplier(step, warmup, config['cycle_steps'], config['minimum_lr_fraction'])
        for group in optimizer.param_groups:
            group['lr'] = recipe['learning_rate'] * group['lr_multiplier'] * multiplier
        scale_optimizer.param_groups[0]['lr'] = config['scale_learning_rate'] * multiplier
        nr = int(config['batch_size'] * config['root_fraction'])
        indices = torch.cat([root_indices[torch.randint(len(root_indices), (nr,), generator=generator)],
                             torch.randint(size, (config['batch_size']-nr,), generator=generator)]).to(device)
        batch = base.core.batch_at(data['train'], indices)
        optimizer.zero_grad(set_to_none=True)
        scale_optimizer.zero_grad(set_to_none=True)
        rng = torch.cuda.get_rng_state(device)

        def backward_loss():
            if config['compile']:
                torch.compiler.cudagraph_mark_step_begin()
            with torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
                output = trained(**batch)
                loss = F.cross_entropy(output.logits.float(), batch['labels'])
            loss.backward()
            return float(loss.detach())

        first_loss = backward_loss()
        with sam.sharpness_perturbation(ordinary, config['sam_radius']) as first_norm:
            optimizer.zero_grad(set_to_none=True)
            scale_optimizer.zero_grad(set_to_none=True)
            torch.cuda.set_rng_state(rng, device)
            sharp_loss = backward_loss()
            if step == 1:
                validate_scale_records(collect_attention_scaling_stats(model), alpha, config['cap'])
                with torch.no_grad():
                    perturbed_max = max(float(m.weight.float().norm()) for _, m in target_weight_modules(model))
                assert perturbed_max <= config['cap'] * (1 + 1e-6)
                save_json(args.output / 'perturbed_bounds.json', {'maximum_effective_weight_norm': perturbed_max,
                          'radius': config['sam_radius'], 'floor_enforced': True})
        norm = float(torch.nn.utils.clip_grad_norm_(ordinary, 1., error_if_nonfinite=True))
        torch.nn.utils.clip_grad_norm_(scales, 1., error_if_nonfinite=True)
        optimizer.step()
        scale_optimizer.step()
        project_attention_scales_(model, scale_optimizer)
        if step == 1:
            save_json(args.output / 'first_step_gradient_audit.json', base.dense.gradient_audit(model, before))
            del before
        completed = step
        training.append({'step': step, 'mean_training_loss': first_loss, 'perturbed_loss': sharp_loss,
                         'first_gradient_norm': first_norm, 'gradient_norm_before_clip': norm,
                         'top_learning_rate': recipe['learning_rate']*multiplier,
                         'step_seconds': time.monotonic()-tick})
        if step % config['log_every'] == 0 or step == 1:
            save_csv(args.output / 'training.csv', training)
            save_json(args.output / 'progress.json', {'status': 'training', 'step': step,
                      'maximum_steps': recipe['steps'], 'best_correct': best_scores['correct']})
            print(f'seed={args.seed} train_step={step} loss={first_loss:.4f} sharp_loss={sharp_loss:.4f}', flush=True)
        if step % config['eval_every'] == 0 or step == recipe['steps']:
            if evaluate(step) and not args.smoke:
                break
    save_csv(args.output / 'training.csv', training)
    checkpoint_path = args.output / 'selected.pth'
    if not args.smoke:
        torch.save({'model_state_dict': best_state, 'step': best_step, 'selected_variant': 'live',
                    'selection_metrics': best_scores, 'model': name, 'mode': 'bounded',
                    'seed': args.seed, 'config': config}, args.output / 'selected.tmp.pth')
        os.replace(args.output / 'selected.tmp.pth', checkpoint_path)
    result = {'name': name, 'model': recipe['model_id'], 'mode': 'bounded', 'seed': args.seed,
        'selected_step': best_step, 'selected_variant': 'live', 'selection_accuracy': best_key[0],
        'evaluation': best_scores, 'historical_baseline_target': recipe['baseline_target'],
        'historical_recovery_threshold': recipe['baseline_target']-.01,
        'meets_historical_threshold': best_scores['accuracy'] >= recipe['baseline_target']-.01,
        'alpha_s': alpha, 'alpha_w': config['cap'], 'configured_denominator_floor': alpha**2*config['cap']**2,
        'maximum_observed_matrix_to_alpha_ratio': max(row['maximum_matrix_to_alpha_ratio'] for row in norm_rows),
        'applied_caps': caps, 'adapters': adapters, 'seconds': time.monotonic()-started, 'smoke': args.smoke,
        'checkpoint_sha256': base.dense.file_sha256(checkpoint_path) if not args.smoke else None,
        'selection_protocol': 'official_validation', 'training_sampling_protocol': config['training_sampling_protocol'],
        'initialization': metadata['initialization'], 'parent_checkpoint': config['parent_checkpoint'],
        'seed_role': metadata['seed_role'], 'train_scope': metadata['train_scope'],
        'trainable_parameters': sum(p.numel() for p in model.parameters()), 'optimizer_steps': completed,
        'training_examples_seen': completed*config['batch_size'], 'external_teacher_used': False}
    save_json(args.output / 'result.json', result)
    save_json(args.output / 'manifest.json', metadata)
    save_json(args.output / 'progress.json', {'status': 'complete', 'optimizer_steps': completed,
              'selected_step': best_step, 'evaluation': best_scores})
    print(json.dumps({'seed': args.seed, 'evaluation': best_scores, 'recovered': result['meets_historical_threshold']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--smoke', action='store_true')
    main(parser.parse_args())

