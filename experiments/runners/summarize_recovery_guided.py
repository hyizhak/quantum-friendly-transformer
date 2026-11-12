"""Summarize all disclosed guided-protocol attempts, including failed controls."""

import argparse
import csv
import json
import math
from pathlib import Path


def compare_predictions(before, after):
    def read(path):
        with path.open() as handle:
            return {int(row['example_id']): row for row in csv.DictReader(handle)}
    original, selected = read(before), read(after)
    assert original.keys() == selected.keys()
    fixed, regressed, changed = 0, 0, 0
    for identifier, row in original.items():
        other = selected[identifier]
        assert row['label'] == other['label']
        old_correct = row['prediction'] == row['label']
        new_correct = other['prediction'] == other['label']
        fixed += not old_correct and new_correct
        regressed += old_correct and not new_correct
        changed += row['prediction'] != other['prediction']
    return {'fixed_previous_errors': fixed, 'introduced_errors': regressed,
            'changed_predictions': changed, 'examples': len(original)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    args = parser.parse_args()
    root = args.results
    cohorts = ['recovery_trajectory', 'recovery_guided', 'recovery_classifier_fit', 'recovery_guided_restart']
    parents = {'2layers33m': 'recovery_2layer_seed_extension/2layers33m_bounded_seed17',
               'olmo1b': 'recovery_head/olmo1b_bounded_seed151',
               'mistral7b': 'recovery_full_dense/mistral7b_bounded_seed401'}
    attempts = []
    for cohort in cohorts:
        for path in sorted((root / cohort).glob('*_bounded_seed*/result.json')):
            r = json.loads(path.read_text())
            if r['smoke']:
                continue
            attempts.append({'model': r['name'], 'cohort': cohort, 'seed': r['seed'],
                'correct': r['evaluation']['correct'], 'examples': r['evaluation']['examples'],
                'accuracy': r['evaluation']['correct'] / r['evaluation']['examples'],
                'mean_nll': r['evaluation']['mean_nll'], 'selected_step': r['selected_step'],
                'selected_variant': r.get('selected_variant', 'classifier candidate'),
                'qualifies': r['meets_historical_threshold'], 'teacher_used': r.get('external_teacher_used', False),
                'optimizer_steps': r.get('optimizer_steps'), 'classifier_candidates': r.get('candidates_fitted'),
                'seed_role': r.get('seed_role'), 'result_path': str(path)})
    rows = []
    for name, relative in parents.items():
        prior = json.loads((root / relative / 'result.json').read_text())
        relevant = [r for r in attempts if r['model'] == name]
        best = max(relevant, key=lambda r: (r['accuracy'], -r['mean_nll'])) if relevant else None
        previous_correct = prior['evaluation']['correct']
        best_correct = max(previous_correct, best['correct'] if best else 0)
        examples = prior['evaluation']['examples']
        required_correct = math.ceil(prior['historical_recovery_threshold'] * examples)
        current_nll = prior['evaluation']['mean_nll']
        selected_predictions = root / relative / 'predictions.csv'
        if best and (best['correct'], -best['mean_nll']) > (previous_correct, -current_nll):
            current_nll = best['mean_nll']
            selected_predictions = Path(best['result_path']).with_name('predictions.csv')
        rows.append({'model': name, 'previous_correct': previous_correct, 'current_best_correct': best_correct,
                     'examples': examples, 'current_best_accuracy': best_correct / examples,
                     'additional_correct': best_correct - previous_correct,
                     'threshold': prior['historical_recovery_threshold'],
                     'required_correct': required_correct, 'shortfall_examples': max(0, required_correct - best_correct),
                     'qualifies': best_correct / examples >= prior['historical_recovery_threshold'],
                     'previous_mean_nll': prior['evaluation']['mean_nll'],
                     'current_best_mean_nll': current_nll,
                     'paired_validation_changes': compare_predictions(root / relative / 'predictions.csv', selected_predictions),
                     'best_new_attempt': best, 'completed_new_attempts': len(relevant)})
    report = {'rows': rows, 'attempts': attempts,
              'teacher': json.loads((root / 'recovery_teacher/manifest.json').read_text())['teacher_evaluation'],
              'scope': 'Only previously unrecovered models. Original inference architectures, cap24, and historical attention floors retained.',
              'interpretation': 'Selected validation examples, not independent test results or unbiased seed averages. External teacher is an added training resource.'}
    (root / 'recovery_guided_summary.json').write_text(json.dumps(report, indent=2) + '\n')
    with (root / 'recovery_guided_attempts.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(attempts[0]))
        writer.writeheader()
        writer.writerows(attempts)
    lines = ['GUIDED TRAINING PROTOCOL RESULTS', '',
             'Model            Previous best   Current best    Correct / required   Remaining',
             *[f"{r['model']:16s} {100*r['previous_correct']/r['examples']:8.2f}%       {100*r['current_best_accuracy']:8.2f}%       {r['current_best_correct']:3d} / {r['required_correct']:3d}             {r['shortfall_examples']:3d}" for r in rows], '',
             f'Completed new attempts in these cohorts: {len(attempts)}. Every attempt is listed in recovery_guided_attempts.csv.',
             f'Newly recovered models in this round: {sum(r["qualifies"] for r in rows)}.', '',
             'Mean NLL for the best accuracy, breaking ties by lower NLL:',
             *[f"  {r['model']}: {r['previous_mean_nll']:.6f} -> {r['current_best_mean_nll']:.6f}" for r in rows], '',
             'Paired validation changes relative to the previous selected checkpoint:',
             *[f"  {r['model']}: {r['paired_validation_changes']['fixed_previous_errors']} previous errors fixed; "
               f"{r['paired_validation_changes']['introduced_errors']} new errors introduced." for r in rows], '',
             'The RoBERTa teacher scored 821/872 (94.15%) locally; only original training rows received cached soft targets.',
             'Tested: teacher distillation gated by agreement with training gold labels, cyclic learning rates, layer/head rates,',
             'mixed root/phrase minibatches, within-trajectory parameter averaging, and a deterministic frozen-feature classifier fit.',
             'The two-layer control, teacher continuation, classifier grid, and three bounded pretrained restarts are all retained.',
             'Classifier candidates change only the existing bias-free head. Teacher-guided students use a single unchanged model at inference.',
             'The original caps and floors remain enforced at every student training and evaluation forward.', '',
             'Budgets: 2,000 updates each for the two-layer control and guided continuation; 30 classifier candidates;',
             '5,000 updates for each of three pretrained restarts; 800 OLMo updates; up to 400 Mistral updates.',
             'The selected classifier candidate can have zero teacher coefficient even though the overall grid tests a teacher.',
             'Selected step zero means the continuation retained its original parent, without an accuracy gain.', '',
             'Limits: validation chooses checkpoints, averaging variants, and recipes. The teacher is an external SST-2-finetuned',
             'checkpoint, not our own controlled teacher training. This does not certify full theorem input-matrix assumptions.',
             'Accuracy and negative log likelihood can move in different directions; the attempt CSV includes both.',
             'Recorded trajectories: figures/recovery_guided.pdf (generated after all three models finish).',
             'Verification: recovery_validation_coverage.json and each cohort artifact_validation/checkpoint_replay report.',
             'Method references: https://arxiv.org/abs/1503.02531 and https://arxiv.org/abs/1803.05407',
             'Teacher: https://huggingface.co/textattack/roberta-base-SST-2/tree/84ee248b91053ef5d0c748bbac4edfba1cf89584']
    (root / 'recovery_guided_summary.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines[:9]))


if __name__ == '__main__':
    main()
