"""Count distinct recovered models and disclose every new bounded attempt."""

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

from experiments.lib.reviewer import save_csv, save_json


def summarize(root):
    with (root / 'new_code_audit/recovery_comparison.csv').open() as handle:
        archived = [r for r in csv.DictReader(handle) if r['regime'] == 'bounded']
    attempts, incomplete = [], []
    directories = ['recovery_extension', 'recovery_extension_low_lr', 'recovery_positive_control',
                   'recovery_seed_search', 'recovery_large_search', 'recovery_longer_search',
                   'recovery_gentler_search', 'recovery_large_adaptation', 'recovery_matched_protocol',
                   'recovery_low_rank', 'recovery_length_weighted', 'recovery_roots',
                   'recovery_2layer_seed_extension', 'recovery_curriculum', 'recovery_head',
                   'recovery_average', 'recovery_average_pairs', 'recovery_2layer_full_low_lr',
                   'recovery_full_dense', 'recovery_trajectory', 'recovery_guided',
                   'recovery_classifier_fit', 'recovery_guided_restart', 'recovery_sam']
    for directory in directories:
        for path in sorted((root / directory).glob('*_bounded_seed*')):
            if not (path / 'manifest.json').exists():
                incomplete.append(str(path))
                continue
            result = json.loads((path / 'result.json').read_text())
            if result['smoke']:
                continue
            accuracy = result['evaluation']['correct'] / result['evaluation']['examples']
            attempts.append({'cohort': directory, 'model': result['model'], 'name': result['name'],
                             'seed': result['seed'], 'accuracy': accuracy,
                             'seed_role': result.get('seed_role', 'training order and/or initialization'),
                             'correct': result['evaluation']['correct'],
                             'examples': result['evaluation']['examples'],
                             'threshold': result['historical_recovery_threshold'],
                             'qualifies': result['meets_historical_threshold'],
                             'selected_step': result['selected_step'],
                             'selected_variant': result.get('selected_variant', 'live'),
                             'external_teacher_used': result.get('external_teacher_used', False),
                             'selection_protocol': result.get('selection_protocol', 'internal_training_split'),
                             'training_sampling_protocol': result.get('training_sampling_protocol', 'all_training_phrases'),
                             'initialization': result.get('initialization', 'original pretrained backbone'),
                             'parent_checkpoint': result.get('parent_checkpoint', {}).get('path'),
                             'maximum_observed_matrix_to_alpha_ratio': result['maximum_observed_matrix_to_alpha_ratio'],
                             'result_path': str(path / 'result.json')})
    rows = []
    for old in archived:
        runs = [r for r in attempts if r['model'] == old['model']]
        best = max(runs, key=lambda r: r['accuracy']) if runs else None
        old_recovered = old['recovered'] == 'True'
        new_recovered = any(r['qualifies'] for r in runs)
        rows.append({'model': old['model'], 'baseline': float(old['baseline_accuracy']),
                     'threshold': float(old['recovery_threshold']),
                     'archived_best_accuracy': float(old['best_validation_accuracy']),
                     'archived_recovered': old_recovered,
                     'new_best_accuracy': best['accuracy'] if best else None,
                     'new_best_seed': best['seed'] if best else None,
                     'new_best_cohort': best['cohort'] if best else None,
                     'new_best_selection_protocol': best['selection_protocol'] if best else None,
                     'new_best_training_sampling_protocol': best['training_sampling_protocol'] if best else None,
                     'new_best_seed_role': best['seed_role'] if best else None,
                     'new_best_initialization': best['initialization'] if best else None,
                     'new_best_external_teacher_used': best['external_teacher_used'] if best else False,
                     'new_qualifying_attempts': sum(r['qualifies'] for r in runs),
                     'new_completed_attempts': len(runs), 'new_recovered': new_recovered,
                     'recovered_in_either': old_recovered or new_recovered,
                     'newly_added_model': new_recovered and not old_recovered,
                     'new_best_result_path': best['result_path'] if best else None})
    recovered = sum(r['recovered_in_either'] for r in rows)
    save_csv(root / 'recovery_story.csv', rows)
    save_csv(root / 'recovery_story_all_attempts.csv', attempts)
    save_json(root / 'recovery_story.json', {'distinct_models_recovered': recovered, 'total_models': len(rows),
              'rows': rows, 'all_completed_bounded_attempts': attempts, 'incomplete_attempt_directories': incomplete,
              'interpretation': 'Selected successful examples under configured scalar floors; not an unbiased seed average or full theorem certification.'})
    lines = [f'CONFIGURED-FLOOR RECOVERY: {recovered}/{len(rows)} DISTINCT MODELS', '',
             'Counts combine archived bounded results with new completed experiments.',
             'Each model counts once. Repeating a prior success does not add a model.', '',
             'Model                         Threshold  Archived best  New best  New passes/attempts  Recovered']
    for r in rows:
        name = r['model'].split('/')[-1]
        new = f"{100*r['new_best_accuracy']:.2f}" if r['new_best_accuracy'] is not None else '--'
        lines.append(f"{name:29s} {100*r['threshold']:8.2f} {100*r['archived_best_accuracy']:13.2f} {new:9s} "
                     f"{r['new_qualifying_attempts']}/{r['new_completed_attempts']:3d}               {r['recovered_in_either']}")
    lines += ['', 'Newly recovered distinct models: ' + ', '.join(r['model'] for r in rows if r['newly_added_model']),
              f'All completed new bounded attempts retained: {len(attempts)}.',
              f'Incomplete attempt directories (not counted): {len(incomplete)}.', '',
              'Reporting limits:',
              '- New best values and successful seeds are selected exploratory examples. They do not estimate average-seed performance.',
              '- The fixed cohorts retain every seed and their matched uniform controls in recovery_extension_comparison.txt.',
              '- Standard extension/search runs select checkpoints on a training-data split. Matched-protocol, length-weighted, and training-root runs explicitly use official-validation checkpoint selection. Their training-data policies are recorded per attempt.',
              '- Subsequent seed/recipe selection uses official validation and is exploratory. Validation-selected scores are not independent test scores.',
              '- Curriculum attempts continue a recorded bounded parent with a fresh optimizer. Their seed controls training order; these are not independent pretrained initializations.',
              '- Full dense Mistral fine-tuning materializes the selected adapter parent into FP32 dense weights and trains all model parameters with an 8-bit AdamW state. The caps and historical alpha floor remain fixed; this is a changed optimization recipe.',
              '- Parameter-average candidates combine compatible bounded checkpoints into one model with no additional training. Their numeric identifiers are not training seeds.',
              '- Trajectory-average continuations train from a selected parent and evaluate live and within-run averaged parameters. Teacher-guided runs additionally use soft targets from a pinned external SST-2 teacher on training examples only; this changes training resources and is disclosed per attempt.',
              '- SAM continuations use two gold-label gradient passes per minibatch, perturb and restore ordinary parameters, and retain the original caps and floors. Their recipes and validation-selected results remain exploratory.',
              '- The fixed alpha denominator floors are enforced. Historical alpha_s constants are not certified bounds on full attention-input matrices; exceedances are recorded.',
              '- Native normalization, residual paths, and attention architecture are retained. These are not fully theorem-certified models.',
              '- Short-sentence recovery alone does not establish selective routing, long-context performance, or quantum advantage.',
              '- Archived and new protocols differ; this is evidence that a qualifying example exists, not a controlled estimate of improvement.', '',
              'Provenance: recovery_story_all_attempts.csv links every completed new bounded result; each run saves its config, source/asset hashes, predictions, diagnostics, and selected checkpoint.']
    (root / 'recovery_story.txt').write_text('\n'.join(lines) + '\n')
    tex = [r'\begin{table}[t]', r'\centering', r'\begin{tabular}{lrrrr}', r'\hline',
           r'Model & Threshold (\%) & Archived best (\%) & New best (\%) & Recovered \\', r'\hline']
    for r in rows:
        new = f"{100*r['new_best_accuracy']:.2f}" if r['new_best_accuracy'] is not None else '--'
        marks = r'\dagger' if r['new_best_selection_protocol'] == 'official_validation' else ''
        if r['new_best_seed_role'] == 'candidate identifier; no new training seed':
            marks += r'\ddagger'
        if r['new_best_external_teacher_used']:
            marks += r'\S'
        if marks:
            new = '$' + new + '^{' + marks + '}$'
        tex.append(f"{r['model'].split('/')[-1]} & {100*r['threshold']:.2f} & {100*r['archived_best_accuracy']:.2f} & {new} & "
                   + ('Yes' if r['recovered_in_either'] else 'No') + r' \\')
    tex += [r'\hline', r'\end{tabular}',
            r'\caption{Selected SST-2 recovery examples under configured alpha denominator floors. Each model is counted once across archived and new evidence. New best values are selected across disclosed seeds and exploratory recipes, not unbiased seed means. A dagger denotes official-validation selection; its score is not an independent test score. A double dagger denotes a single-model parameter average of bounded checkpoints, with no additional training. A section sign marks experiments using an external SST-2 teacher during training. Other new runs use internal training-split checkpoint selection. Training-data sampling and every attempt are disclosed separately. Recovery means reaching historical native accuracy minus one percentage point. The supplied alpha constants do not certify the full attention-input matrix bound, so these experiments do not establish all theorem assumptions.}',
            r'\label{tab:bounded-recovery-selected}', r'\end{table}']
    (root / 'recovery_story_table.tex').write_text('\n'.join(tex) + '\n')
    added = ', '.join(r['model'].split('/')[-1] for r in rows if r['newly_added_model'])
    addendum = [
        '% Generated from the complete disclosed recovery ledger; original response.tex is unchanged.',
        'We have added SST-2 recovery experiments with explicitly enforced attention-denominator floors '
        r'$\alpha_{0,\ell}\geq\alpha_{s,\ell}^{2}\alpha_w^{2}$ and Frobenius caps '
        r'$\alpha_w=24$ on the recorded attention and feed-forward weight matrices. '
        f'Across the archived and new evidence, {recovered} of {len(rows)} distinct models have at least one '
        'validation result within one percentage point below their historical native-scaling baseline. '
        f'The newly added recovered models are {added}. '
        'Recovery is an operational threshold comparison, not a statistical equivalence test.', '',
        'We retain every completed attempt and disclose the adaptive choice of seeds, optimization recipes, '
        'training-data sampling, and parent checkpoints. These are selected recovery examples. '
        'The accompanying table marks new runs that select checkpoints on official validation; '
        'their scores are not independent test estimates. Repeated successes of the same model are counted once.', '',
        'The experiments retain native normalization and architecture. The supplied input constants '
        r'$\alpha_{s,\ell}$ are not certified bounds on complete attention-input matrices, and matrix-norm '
        'exceedances are recorded. These experiments establish classification recovery under the configured '
        'scalar restrictions for the qualifying models. They do not validate all theorem assumptions, '
        'long-context content routing, or a quantum advantage.', '',
        r'\input{results/reviewer/recovery_story_table.tex}',
    ]
    full_dense = [attempt for attempt in attempts if attempt['cohort'] == 'recovery_full_dense']
    if full_dense:
        chosen = max(full_dense, key=lambda attempt: attempt['accuracy'])
        dense_result = json.loads(Path(chosen['result_path']).read_text())
        dense_config = json.loads(Path(chosen['result_path']).with_name('config.json').read_text())
        parent = json.loads(Path(dense_config['parent_checkpoint']['result_path']).read_text())
        outcome = 'meets' if chosen['qualifies'] else 'remains below'
        addendum[5:5] = [
            'We also extended Mistral-7B from bounded adapter tuning to full dense fine-tuning: '
            f"all {dense_result['trainable_parameters'] / 1e9:.2f} billion parameters were trainable, "
            'with nonzero gradients verified in all 224 attention and feed-forward projection matrices. '
            'The selected adapter checkpoint was materialized into ordinary dense matrices, retaining '
            'the configured caps and attention floor. '
            f"After {dense_result['optimizer_steps']} updates, the best validation accuracy was "
            f"{100 * chosen['accuracy']:.2f}\\% ({chosen['correct']}/872), compared with "
            f"{100 * parent['evaluation']['accuracy']:.2f}\\% for its selected parent. "
            f"This {outcome} the {100 * chosen['threshold']:.2f}\\% recovery threshold. "
            'The run uses FP32 master weights, BF16 computation, and 8-bit AdamW state; '
            'it is an exploratory continuation with validation checkpoint selection, not a controlled '
            'estimate of the effect of unfreezing alone.', '',
        ]
    guided_cohorts = {'recovery_trajectory', 'recovery_guided', 'recovery_classifier_fit', 'recovery_guided_restart'}
    guided_attempts = [attempt for attempt in attempts if attempt['cohort'] in guided_cohorts]
    guided_names = ['2layers33m', 'olmo1b', 'mistral7b']
    if all(any(attempt['name'] == name for attempt in guided_attempts) for name in guided_names):
        guided_best = {name: max((attempt for attempt in guided_attempts if attempt['name'] == name),
                                key=lambda attempt: attempt['accuracy']) for name in guided_names}
        guided_recovered = sum(attempt['qualifies'] for attempt in guided_best.values())
        scores = ', '.join(f"{100 * guided_best[name]['accuracy']:.2f}\\%" for name in guided_names)
        addendum[-1:-1] = [
            f'An additional {len(guided_attempts)} exploratory protocol runs were restricted to the three '
            'models still missing recovery. We tested training-label-gated teacher distillation, cyclic '
            'learning rates, layer-specific rates, sentence/phrase sampling, trajectory parameter averages, '
            'and a 30-candidate frozen-feature classifier fit. The teacher is an externally fine-tuned '
            'RoBERTa SST-2 checkpoint; only its logits on original training rows enter student losses, '
            'and it is an additional training resource. Best accuracies among these attempts were '
            f'{scores} for TinyStories-2Layers-33M, OLMo-1B, and Mistral-7B, respectively; '
            f'{guided_recovered} additional models reached their original recovery thresholds. '
            'The specified caps, denominator floors, and inference architectures were retained. '
            'All failed controls and restarts are included in the attempt ledger. These results use '
            'adaptive validation selection and do not isolate the effect of each protocol component.', '',
        ]
    sam_attempts = [attempt for attempt in attempts if attempt['cohort'] == 'recovery_sam']
    if sam_attempts:
        chosen = max(sam_attempts, key=lambda attempt: attempt['accuracy'])
        sam_result = json.loads(Path(chosen['result_path']).read_text())
        parent = json.loads(Path(sam_result['parent_checkpoint']['result_path']).read_text())
        outcome = 'reaches' if chosen['qualifies'] else 'remains below'
        addendum[-1:-1] = [
            f'We subsequently tested {len(sam_attempts)} sharpness-aware full-parameter continuations '
            'of TinyStories-2Layers-33M using gold training labels and the same configured restrictions. '
            f"The best selected checkpoint scores {chosen['correct']}/872 "
            f"({100 * chosen['accuracy']:.2f}\\%), compared with {parent['evaluation']['correct']}/872 "
            f"for its parent, and {outcome} the original {100 * chosen['threshold']:.2f}\\% threshold. "
            'The predeclared sequence stops after the first recovery and retains all preceding failures. '
            'This supplies another selected recovery example; it does not establish a population-level '
            'performance improvement or certify the full attention-input matrix bound.', '',
        ]
    (root / 'recovery_response_addendum.tex').write_text('\n'.join(addendum) + '\n')
    print('\n'.join(lines[:len(rows) + 6]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=ROOT / 'results/reviewer')
    summarize(parser.parse_args().results)
