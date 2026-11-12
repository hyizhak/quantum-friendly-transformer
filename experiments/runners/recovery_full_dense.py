"""Full dense Mistral fine-tuning under the unchanged configured bounds."""

import argparse
import copy
import csv
import gc
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

import torch

from experiments.lib.experiment import load_config
from experiments.lib import recovery_full_dense as dense
from experiments.lib import recovery_root_data as roots
from experiments.lib.recovery_extension import validate_scale_records
from experiments.lib.experiment import manifest, save_csv, save_json
from experiments.runners import recovery_extension as core
from experiments.runners import recovery_large as large
from experiments.runners import recovery_low_rank as low_rank
from quantum_friendly_transformer.norm_transformer import (
    collect_attention_scaling_stats, collect_learnable_cap_stats, project_attention_scales_,
)
from quantum_friendly_transformer.norm_transformer.frobenius_cap import target_weight_modules


def build_model(assets, config, seed, devices, initialize_parent=True):
    if initialize_parent:
        parent_config = copy.deepcopy(config)
        parent_config['models']['mistral7b'].update(master_dtype='bfloat16', train_scope='head_caps')
        model, caps, adapters = low_rank.build_model(assets, 'mistral7b', parent_config,
                                                    seed, 'bounded', torch.device('cpu'))
        dense.load_verified_parent(model, config)
        conversion = dense.materialize_dense(model, torch.device(devices[0]))
    else:
        model, caps, adapters = large.build_model(assets, 'mistral7b', config, seed,
                                                 'bounded', torch.device('cpu'))
        model.backbone.embed_tokens.register_forward_hook(dense.bf16_embedding_output)
        conversion = []
    mapping = dense.place_model(model, devices)
    return model, caps, adapters, conversion, mapping


def make_optimizers(model, config):
    import bitsandbytes as bnb
    assert bnb.__version__ == config['optimizer_version']
    ordinary, scales = [], []
    for name, parameter in model.named_parameters():
        (scales if '_qft_learnable_attention_scale.' in name else ordinary).append(parameter)
    main = bnb.optim.AdamW8bit(ordinary, lr=config['models']['mistral7b']['learning_rate'],
                             weight_decay=config['weight_decay'], eps=1e-8)
    controller = torch.optim.AdamW(scales, lr=config['scale_learning_rate'],
                                  weight_decay=0., eps=1e-12, foreach=False)
    covered = [id(p) for opt in [main, controller] for group in opt.param_groups for p in group['params']]
    assert len(covered) == len(set(covered))
    assert set(covered) == {id(p) for p in model.parameters()}
    return main, controller, ordinary, scales


def memory_stats(devices):
    return [{'device': str(device), 'allocated_gib': torch.cuda.memory_allocated(device) / 2**30,
             'peak_allocated_gib': torch.cuda.max_memory_allocated(device) / 2**30,
             'reserved_gib': torch.cuda.memory_reserved(device) / 2**30} for device in devices]


def main(args):
    config = load_config(args.config, getattr(args, "checkpoint_root", None))
    if args.smoke:
        config['models']['mistral7b']['steps'] = 2
        config['eval_every'] = 2
        config['log_every'] = 1
    args.output.mkdir(parents=True, exist_ok=False)
    save_json(args.output / 'config.json', config)
    torch.set_num_threads(config['threads'])
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    devices = [torch.device(d) for d in args.devices]
    assert len(devices) == 2 and all(d.type == 'cuda' for d in devices)
    alpha = config['models']['mistral7b']['alpha_s']
    started = time.monotonic()
    print('Preparing pinned training data and materializing the bounded parent', flush=True)
    tokenizer, data, split = roots.prepare_data(args.assets, 'mistral7b', config)
    save_json(args.output / 'split.json', split)
    if args.smoke:
        data = {name: {k: v[:32] for k, v in batch.items()} for name, batch in data.items()}
    data = {name: {k: v.to(devices[0]) for k, v in batch.items()} for name, batch in data.items()}
    model, caps, adapters, conversion, mapping = build_model(args.assets, config, args.seed, devices)
    model.config.pad_token_id = tokenizer.pad_token_id
    save_json(args.output / 'conversion.json', conversion)
    inventory = dense.parameter_inventory(model)
    save_json(args.output / 'parameter_inventory.json', inventory)
    save_json(args.output / 'device_map.json', mapping)
    print(f'All {inventory["trainable_parameters"]:,} parameters trainable; dense matrices on both GPUs', flush=True)
    sources = [Path(__file__), Path(dense.__file__), Path(core.__file__), Path(large.__file__),
               Path(low_rank.__file__), Path(roots.__file__), roots.ALIGNMENT,
               ROOT / 'experiments/lib/bounded_low_rank.py', ROOT / 'experiments/lib/recovery_extension.py',
               ROOT / 'experiments/lib/experiment.py', ROOT / 'experiments/runners/recovery_matched.py',
               ROOT / 'src/quantum_friendly_transformer/norm_transformer/frobenius_cap.py',
               ROOT / 'src/quantum_friendly_transformer/norm_transformer/attention_scaling.py',
               ROOT / 'src/quantum_friendly_transformer/experiments/runners/recovery/glue_sequence_classifier.py']
    metadata = manifest(config, sources, model='mistral7b', seed=args.seed, mode='bounded',
                        assets=json.loads((args.assets / 'assets.json').read_text()),
                        devices=[str(d) for d in devices], gpu=[torch.cuda.get_device_name(d) for d in devices],
                        dtype='FP32 master parameters and gradients; BF16 embedding outputs and autocast',
                        optimizer='bitsandbytes AdamW8bit; FP32 AdamW for bounded scale controllers',
                        train_scope='all dense model parameters', gradient_checkpointing=True,
                        selection_protocol='official_validation', training_sampling_protocol='training_roots',
                        initialization='materialized effective matrices from selected bounded adapter parent',
                        parent_checkpoint=config['parent_checkpoint'], smoke=args.smoke,
                        limitations=['Official validation selects parent and new checkpoints; these are exploratory selected examples, not independent test scores.',
                                     'Historical scalar alpha floor, not a certified full attention-input matrix bound.',
                                     'Native normalization and architecture remain; this does not validate all theorem assumptions.'])
    save_json(args.output / 'running_manifest.json', metadata)
    core.AttentionInputMonitor = large.LargeAttentionInputMonitor
    selections, training, norm_rows, scale_rows, weight_rows = [], [], [], [], []
    best = (-math.inf, -math.inf)
    selected_step = None
    selected_path = args.output / 'selected.pth'

    def evaluate_and_select(step):
        nonlocal best, selected_step
        scores, predictions, losses, norms = core.evaluate(model, data['selection'],
            config['eval_batch_size'], alpha, devices[0], monitor=True)
        scales = collect_attention_scaling_stats(model)
        validate_scale_records(scales, alpha, config['cap'])
        weights = collect_learnable_cap_stats(model)
        # Audit the actual dense matrices, independently of gamma statistics.
        with torch.no_grad():
            maximum_weight_norm = max(float(m.weight.float().norm()) for _, m in target_weight_modules(model))
        assert maximum_weight_norm <= config['cap'] * (1 + 1e-6)
        selections.append({'step': step, 'seconds': time.monotonic() - started,
                           'maximum_effective_weight_norm': maximum_weight_norm, **scores})
        norm_rows.extend({'step': step, **row} for row in norms)
        scale_rows.extend({'step': step, **row} for row in scales)
        weight_rows.extend({'step': step, **row} for row in weights)
        if step == 0:
            with (Path(config['parent_checkpoint']['path']).parent / 'predictions.csv').open() as handle:
                parent_rows = list(csv.DictReader(handle))[:len(predictions)]
            expected = torch.tensor([int(row['prediction']) for row in parent_rows])
            assert [int(row['example_id']) for row in parent_rows] == data['selection']['example_id'].cpu().tolist()
            save_json(args.output / 'parent_conversion_replay.json', {
                'examples': len(expected), 'prediction_disagreements': int((expected != predictions).sum()),
                'converted_initial_metrics': scores,
                'maximum_nll_difference': max(abs(float(row['nll']) - float(loss)) for row, loss in zip(parent_rows, losses)),
                'note': 'FP32 master matrices materialize the rounded effective adapter weights; BF16 residual precision retained.'})
        key = (scores['accuracy'], -scores['mean_nll'])
        if key > best:
            best, selected_step = key, step
            if not args.smoke:
                checkpoint = {'model_state_dict': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                              'step': step, 'selection_metrics': scores, 'model': 'mistral7b',
                              'mode': 'bounded', 'seed': args.seed, 'config': config}
                temporary = args.output / 'selected.tmp.pth'
                torch.save(checkpoint, temporary)
                os.replace(temporary, selected_path)
                del checkpoint
            save_csv(args.output / 'predictions.csv', [
                {'example_id': int(i), 'label': int(y), 'prediction': int(p), 'nll': float(nll)}
                for i, y, p, nll in zip(data['selection']['example_id'].cpu(),
                                       data['selection']['labels'].cpu(), predictions, losses)])
            save_json(args.output / 'selected_scale_stats.json', scales)
            save_csv(args.output / 'evaluation_attention_inputs.csv', norms)
        for name, rows in [('selection', selections), ('attention_inputs', norm_rows),
                           ('scales', scale_rows), ('weights', weight_rows)]:
            save_csv(args.output / f'{name}.csv', rows)
        print(f'step={step} validation={scores["correct"]}/{scores["examples"]} '
              f'best_step={selected_step} elapsed={time.monotonic()-started:.1f}s', flush=True)
        model.train()

    evaluate_and_select(0)
    optimizer, scale_optimizer, ordinary, scale_parameters = make_optimizers(model, config)
    before = dense.parameter_probes(model)
    generator = torch.Generator().manual_seed(args.seed + 8917)
    size = len(data['train']['labels'])
    order, cursor = torch.randperm(size, generator=generator).to(devices[0]), 0
    steps = config['models']['mistral7b']['steps']
    warmup = max(1, int(steps * config['warmup_fraction']))
    microbatch = config['micro_batch_size']
    accumulation = config['gradient_accumulation_steps']
    assert config['batch_size'] == microbatch * accumulation
    loop_start = time.monotonic()
    for step in range(1, steps + 1):
        step_start = time.monotonic()
        multiplier = step / warmup if step <= warmup else .5 * (1 + math.cos(math.pi * (step - warmup) / max(1, steps - warmup)))
        optimizer.param_groups[0]['lr'] = config['models']['mistral7b']['learning_rate'] * multiplier
        scale_optimizer.param_groups[0]['lr'] = config['scale_learning_rate'] * multiplier
        optimizer.zero_grad(set_to_none=True)
        scale_optimizer.zero_grad(set_to_none=True)
        batch_loss, batch_correct = 0., 0
        for _ in range(accumulation):
            if cursor + microbatch > size:
                order, cursor = torch.randperm(size, generator=generator).to(devices[0]), 0
            indices = order[cursor:cursor + microbatch]
            cursor += microbatch
            batch = core.batch_at(data['train'], indices)
            with torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
                output = model(**batch)
            (output.loss / accumulation).backward()
            batch_loss += float(output.loss.detach()) / accumulation
            batch_correct += int((output.logits.detach().argmax(-1) == batch['labels']).sum())
            del output
        grad_norm = float(torch.nn.utils.clip_grad_norm_(ordinary, 1., error_if_nonfinite=True))
        scale_norm = float(torch.nn.utils.clip_grad_norm_(scale_parameters, 1., error_if_nonfinite=True))
        optimizer.step()
        scale_optimizer.step()
        project_attention_scales_(model, scale_optimizer)
        if step == 1:
            save_json(args.output / 'first_step_gradient_audit.json', dense.gradient_audit(model, before))
            del before
            print('First-step gradient coverage verified for every parameter', flush=True)
        record = {'step': step, 'mean_training_loss': batch_loss,
                  'training_accuracy': batch_correct / config['batch_size'],
                  'learning_rate': optimizer.param_groups[0]['lr'], 'gradient_norm_before_clip': grad_norm,
                  'scale_gradient_norm_before_clip': scale_norm, 'step_seconds': time.monotonic() - step_start,
                  'seconds': time.monotonic() - loop_start}
        training.append(record)
        save_csv(args.output / 'training.csv', training)
        save_json(args.output / 'progress.json', {'status': 'training', **record,
                  'total_steps': steps, 'selected_step': selected_step, 'selected_accuracy': best[0],
                  'memory': memory_stats(devices)})
        if step % config['log_every'] == 0 or step == 1:
            print(f'train_step={step} loss={batch_loss:.4f} seconds_per_step={record["step_seconds"]:.2f}', flush=True)
        if step % config['eval_every'] == 0 or step == steps:
            optimizer.zero_grad(set_to_none=True)
            scale_optimizer.zero_grad(set_to_none=True)
            evaluate_and_select(step)
    selected = next(row for row in selections if row['step'] == selected_step)
    scores = {k: selected[k] for k in ['accuracy', 'correct', 'examples', 'mean_nll']}
    recipe = config['models']['mistral7b']
    result = {'model': recipe['model_id'], 'name': 'mistral7b', 'mode': 'bounded', 'seed': args.seed,
              'selected_step': selected_step, 'selection_accuracy': best[0], 'evaluation': scores,
              'historical_baseline_target': recipe['baseline_target'],
              'historical_recovery_threshold': recipe['baseline_target'] - .01,
              'meets_historical_threshold': scores['accuracy'] >= recipe['baseline_target'] - .01,
              'alpha_s': alpha, 'alpha_w': config['cap'], 'configured_denominator_floor': alpha**2 * config['cap']**2,
              'maximum_observed_matrix_to_alpha_ratio': max(r['maximum_matrix_to_alpha_ratio'] for r in norm_rows),
              'applied_caps': caps, 'adapters': adapters, 'seconds': time.monotonic() - started,
              'smoke': args.smoke, 'checkpoint_sha256': dense.file_sha256(selected_path) if not args.smoke else None,
              'selection_protocol': 'official_validation', 'training_sampling_protocol': 'training_roots',
              'initialization': metadata['initialization'], 'parent_checkpoint': config['parent_checkpoint'],
              'train_scope': 'all dense model parameters', 'trainable_parameters': inventory['trainable_parameters'],
              'optimizer_steps': steps, 'training_examples_seen': steps * config['batch_size'],
              'memory': memory_stats(devices)}
    save_json(args.output / 'result.json', result)
    save_json(args.output / 'manifest.json', metadata)
    save_json(args.output / 'progress.json', {'status': 'complete', 'selected_step': selected_step, 'evaluation': scores})
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint-root', type=Path, help='Root for relative parent-checkpoint paths in the recipe')
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=401)
    parser.add_argument('--devices', nargs='+', default=['cuda:0', 'cuda:1'])
    parser.add_argument('--smoke', action='store_true')
    main(parser.parse_args())
