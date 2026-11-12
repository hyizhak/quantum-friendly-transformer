"""Fit the unrecovered two-layer classifier with a full-batch convex objective."""

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

import torch
from experiments.lib import recovery_classifier_fit as solver
from experiments.lib import recovery_guided as guided
from experiments.lib import recovery_full_dense as dense
from experiments.lib import recovery_root_data as roots
from experiments.lib.recovery_extension import validate_scale_records
from experiments.lib.reviewer import manifest, save_csv, save_json
from experiments.runners import reviewer_recovery_extension as core
from experiments.runners import reviewer_recovery_matched as matched
from experiments.runners import reviewer_recovery_guided as continuation
from quantum_friendly_transformer.norm_transformer import collect_attention_scaling_stats, collect_learnable_cap_stats


def main(args):
    config = json.loads(args.config.read_text())
    name = '2layers33m'
    recipe = config['models'][name]
    args.output.mkdir(parents=True, exist_ok=False)
    save_json(args.output / 'config.json', config)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device(args.device)
    tokenizer, data, split = matched.prepare_data(args.assets, name, config)
    save_json(args.output / 'split.json', split)
    teacher_path = args.teacher / 'manifest.json'
    teacher = json.loads(teacher_path.read_text())
    assets = json.loads((args.assets / 'assets.json').read_text())
    assert teacher['dataset_revision'] == config['dataset_revision']
    assert teacher['training_data_sha256'] == assets['sha256']['data/sst2/train-00000-of-00001.parquet']
    assert dense.file_sha256(args.teacher / 'training_logits.pth') == teacher['training_logits_sha256']
    targets = guided.align_training_targets(data['train'], torch.load(args.teacher / 'training_logits.pth', weights_only=True)).to(device)
    save_json(args.output / 'teacher_reference.json', {'manifest_path': str(teacher_path),
        'manifest_sha256': dense.file_sha256(teacher_path), 'training_logits_sha256': teacher['training_logits_sha256'],
        'teacher_id': teacher['teacher_id'], 'teacher_revision': teacher['teacher_revision'],
        'student_training_rows': len(targets), 'validation_targets_used_for_training': False})
    root_ids = set(json.loads(roots.ALIGNMENT.read_text())['glue_training_example_ids'])
    root_mask = torch.tensor([int(i) in root_ids for i in data['train']['example_id']], device=device)
    data = {key: {k: v.to(device) for k, v in rows.items()} for key, rows in data.items()}
    model, caps, adapters = core.build_model(args.assets, name, config, args.seed, 'bounded', device)
    continuation.load_parent(model, config, name)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.requires_grad_(False)
    model.score.requires_grad_(True)
    parent_weight = model.score.weight.detach().clone()
    sources = [Path(__file__), Path(solver.__file__), Path(guided.__file__), Path(dense.__file__),
               Path(core.__file__), Path(matched.__file__), Path(continuation.__file__), roots.ALIGNMENT,
               ROOT / 'experiments/lib/reviewer.py', ROOT / 'experiments/lib/recovery_extension.py',
               ROOT / 'src/quantum_friendly_transformer/norm_transformer/frobenius_cap.py',
               ROOT / 'src/quantum_friendly_transformer/norm_transformer/attention_scaling.py',
               ROOT / 'src/quantum_friendly_transformer/experiments/runners/recovery/glue_sequence_classifier.py']
    metadata = manifest(config, sources, model=name, seed=args.seed, mode='bounded', assets=assets,
        train_scope='existing classifier only', teacher=teacher, smoke=False,
        initialization='selected bounded parent; deterministic full-batch classifier optimization',
        parent_checkpoint=config['parent_checkpoint'], selection_protocol='official_validation',
        training_sampling_protocol='full_batch_weighted_training_phrases',
        limitations=['Validation selects the regularization/root-weight/teacher-weight candidate; this is exploratory selection, not an independent test.',
                     'The teacher is an added external SST-2 training resource. Student inference has the same bias-free classifier and frozen bounded backbone.',
                     'Historical alpha constants do not certify complete attention-input matrices.'])
    save_json(args.output / 'running_manifest.json', metadata)
    started = time.monotonic()
    evaluations, candidates, scales_all, weights_all, norms_all = [], [], [], [], []
    best_key, best_weight, best_scores, best_id = (-float('inf'), -float('inf')), None, None, None

    def evaluate(candidate_id, details):
        nonlocal best_key, best_weight, best_scores, best_id
        scores, predicted, losses, norms = core.evaluate(model, data['selection'], config['eval_batch_size'],
                                                         recipe['alpha_s'], device, monitor=True)
        scales = collect_attention_scaling_stats(model)
        validate_scale_records(scales, recipe['alpha_s'], config['cap'])
        evaluations.append({'step': candidate_id, 'seconds': time.monotonic()-started, **scores, **details})
        scales_all.extend({'step': candidate_id, **r} for r in scales)
        weights_all.extend({'step': candidate_id, **r} for r in collect_learnable_cap_stats(model))
        norms_all.extend({'step': candidate_id, **r} for r in norms)
        prediction_rows = [{'example_id': int(i), 'label': int(y), 'prediction': int(p), 'nll': float(nll)}
            for i, y, p, nll in zip(data['selection']['example_id'].cpu(), data['selection']['labels'].cpu(), predicted, losses)]
        save_csv(args.output / f'candidate_{candidate_id:02d}_predictions.csv', prediction_rows)
        key = (scores['accuracy'], -scores['mean_nll'])
        if key > best_key:
            best_key, best_weight, best_scores, best_id = key, model.score.weight.detach().cpu().clone(), scores, candidate_id
            save_csv(args.output / 'predictions.csv', prediction_rows)
            save_json(args.output / 'selected_scale_stats.json', scales)
            save_csv(args.output / 'evaluation_attention_inputs.csv', norms)
        for table, rows in [('selection', evaluations), ('scales', scales_all), ('weights', weights_all), ('attention_inputs', norms_all)]:
            save_csv(args.output / f'{table}.csv', rows)
        print(f'candidate={candidate_id} score={scores["correct"]}/872 best={best_scores["correct"]} {details}', flush=True)
        return scores['accuracy'] >= recipe['baseline_target'] - .01

    evaluate(0, {'protocol': 'unchanged parent'})
    chunks = []
    handle = model.score.register_forward_pre_hook(lambda _module, inputs: chunks.append(inputs[0].detach().double().clone()))
    model.eval()
    with torch.no_grad():
        for start in range(0, len(targets), 256):
            with torch.autocast('cuda', dtype=torch.bfloat16):
                model(**core.batch_at(data['train'], slice(start, start + 256)))
    handle.remove()
    features = torch.cat(chunks)
    del chunks
    assert len(features) == len(targets)
    print(f'Cached {features.shape} frozen training features; validation features are not supplied to the solver', flush=True)
    for index, settings in enumerate(config['classifier_candidates'], start=1):
        weight, details = solver.fit_classifier(features, data['train']['labels'], targets,
            parent_weight, root_mask, **settings, max_iter=config['solver_max_iter'])
        with torch.no_grad():
            model.score.weight.copy_(weight)
        candidates.append({'candidate_id': index, **details})
        save_json(args.output / 'candidate_fits.json', candidates)
        if evaluate(index, details):
            break
    with torch.no_grad():
        model.score.weight.copy_(best_weight.to(device))
    checkpoint_path = args.output / 'selected.pth'
    torch.save({'model_state_dict': guided.clone_cpu_state(model), 'step': best_id,
                'selection_metrics': best_scores, 'model': name, 'mode': 'bounded', 'seed': args.seed,
                'config': config}, checkpoint_path)
    result = {'name': name, 'model': recipe['model_id'], 'mode': 'bounded', 'seed': args.seed,
              'selected_step': best_id, 'selection_accuracy': best_key[0], 'evaluation': best_scores,
              'historical_baseline_target': recipe['baseline_target'], 'historical_recovery_threshold': recipe['baseline_target']-.01,
              'meets_historical_threshold': best_scores['accuracy'] >= recipe['baseline_target']-.01,
              'alpha_s': recipe['alpha_s'], 'alpha_w': config['cap'], 'configured_denominator_floor': recipe['alpha_s']**2*config['cap']**2,
              'maximum_observed_matrix_to_alpha_ratio': max(r['maximum_matrix_to_alpha_ratio'] for r in norms_all),
              'applied_caps': caps, 'adapters': adapters, 'seconds': time.monotonic()-started, 'smoke': False,
              'checkpoint_sha256': dense.file_sha256(checkpoint_path), 'selection_protocol': 'official_validation',
              'training_sampling_protocol': metadata['training_sampling_protocol'], 'initialization': metadata['initialization'],
              'parent_checkpoint': config['parent_checkpoint'], 'external_teacher_used': True,
              'seed_role': 'experiment identifier; deterministic classifier optimization',
              'candidates_fitted': len(candidates), 'solver_iterations': sum(c['solver_iterations'] for c in candidates)}
    save_json(args.output / 'result.json', result)
    save_json(args.output / 'manifest.json', metadata)
    print(json.dumps({k: result[k] for k in ['evaluation', 'selected_step', 'meets_historical_threshold']}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--teacher', type=Path, default=ROOT / 'results/reviewer/recovery_teacher')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=701)
    parser.add_argument('--device', default='cuda:0')
    main(parser.parse_args())
