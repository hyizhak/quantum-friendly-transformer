"""Report the completed full-parameter Mistral continuation and training curve."""

import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    path = args.run
    result = json.loads((path / 'result.json').read_text())
    assert not result['smoke']
    config = json.loads((path / 'config.json').read_text())
    inventory = json.loads((path / 'parameter_inventory.json').read_text())
    audit = json.loads((path / 'first_step_gradient_audit.json').read_text())
    conversion = json.loads((path / 'parent_conversion_replay.json').read_text())
    selection = list(csv.DictReader((path / 'selection.csv').open()))
    training = list(csv.DictReader((path / 'training.csv').open()))
    parent = json.loads(Path(config['parent_checkpoint']['result_path']).read_text())
    dense = [row for row in audit if '.parametrizations.weight.original' in row['name']]
    update_path = path / 'selected_dense_update_audit.json'
    updates = json.loads(update_path.read_text()) if update_path.exists() else None
    replays = [row for report in path.parent.glob('checkpoint_replay*.json')
               for row in json.loads(report.read_text()).get('reports', [])
               if row['run'] == path.name and row.get('replayed')]
    score = result['evaluation']
    lines = [
        'FULL DENSE MISTRAL FINE-TUNING', '',
        f"Best selected validation: {score['correct']}/{score['examples']} ({100*score['accuracy']:.4f}%) at update {result['selected_step']}.",
        f"Parent checkpoint: {parent['evaluation']['correct']}/872 ({100*parent['evaluation']['accuracy']:.4f}%).",
        f"Recovery threshold: {100*result['historical_recovery_threshold']:.4f}%; recovered: {result['meets_historical_threshold']}.",
        f"Completed {result['optimizer_steps']} optimizer steps, {result['training_examples_seen']} sampled training examples.", '',
        f"Trainable parameters: {inventory['trainable_parameters']:,}/{inventory['total_parameters']:,}.",
        f"First-step finite-gradient coverage: {sum(r['gradient_present'] and r['gradient_finite'] for r in audit)}/{len(audit)} parameter tensors.",
        f"Dense projection matrices with nonzero gradients: {sum(r['gradient_norm'] > 0 for r in dense)}/{len(dense)}.",
        f"Dense projection matrices with detectable sampled updates at the first warmup step: {sum(r['probe_maximum_update'] > 0 for r in dense)}/{len(dense)}.",
        (f"Dense projection matrices verified changed in the selected checkpoint: {updates['matrices_verified_changed']}/{updates['matrix_count']}."
         if updates else 'Selected-checkpoint dense-weight change audit: pending.'),
        f"Converted parent prediction disagreements: {conversion['prediction_disagreements']}/{conversion['examples']}.",
        f"Maximum per-example NLL change during conversion/precision replay: {conversion['maximum_nll_difference']:.6g}.", '',
        (f"Fresh selected-checkpoint replay: all predictions match={replays[0]['all_predictions_match']}; maximum NLL difference={replays[0]['maximum_per_example_nll_difference']}."
         if replays else 'Fresh selected-checkpoint replay: pending.'),
        f"FP32 dense masters/gradients, BF16 activations and matrix operations; AdamW8bit {config['optimizer_version']}.",
        f"Peak LR {config['models']['mistral7b']['learning_rate']}; batch {config['batch_size']}; validation every {config['eval_every']} updates.",
        f"Frobenius cap {result['alpha_w']}; fixed alpha_s {result['alpha_s']}; denominator floor {result['configured_denominator_floor']}.",
        f"Maximum measured effective weight norm: {max(float(r['maximum_effective_weight_norm']) for r in selection):.6f}.",
        (f"Selected raw matrix norms, independent float64 reduction: maximum {updates['maximum_raw_frobenius_float64']:.6f} < cap {result['alpha_w']}."
         if updates and 'maximum_raw_frobenius_float64' in updates else 'Independent float64 raw-norm check: pending.'),
        f"Maximum observed input-matrix norm / historical alpha_s: {result['maximum_observed_matrix_to_alpha_ratio']:.4f}.", '',
        'Scope: Mistral alone expands from bounded adapters to all dense parameters.',
        'TinyStories-2Layers-33M and OLMo already had full-parameter runs. The six recovered models were not retrained.',
        'The two smoke attempts are separately preserved and are not recovery results.',
        'Official validation selects the parent and the new checkpoint; this is an exploratory continuation, not an independent test.',
        'The optimizer and parameterization change together. This is not a controlled estimate of the effect of unfreezing alone.',
        'The configured bounds remain fixed, but historical alpha_s does not certify full attention-input matrices or all theorem assumptions.', '',
        'Artifacts: selected.pth, predictions.csv, selection.csv, training.csv, config.json, manifest.json,',
        'parameter_inventory.json, first_step_gradient_audit.json, parent_conversion_replay.json.',
    ]
    (path / 'summary.txt').write_text('\n'.join(lines) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    steps = [int(row['step']) for row in selection]
    accuracy = [100*float(row['accuracy']) for row in selection]
    axes[0].plot(steps, accuracy, 'o-', color='#156b9a', label='Full dense fine-tuning')
    axes[0].axhline(100*parent['evaluation']['accuracy'], color='#777777', linestyle=':', label='Selected parent')
    axes[0].axhline(100*result['historical_recovery_threshold'], color='#a33b20', linestyle='--', label='Recovery threshold')
    axes[0].set(xlabel='Optimizer update', ylabel='Selected-validation accuracy (%)')
    axes[0].legend(fontsize=8)
    window = 20
    averaged = [(int(training[end-1]['step']), sum(float(r['mean_training_loss']) for r in training[max(0,end-window):end])/min(end,window))
                for end in range(1, len(training)+1)]
    axes[1].plot(*zip(*averaged), color='#156b9a', label='Training NLL, rolling 20 updates')
    axes[1].plot(steps, [float(row['mean_nll']) for row in selection], 'o-', color='#a33b20', label='Validation NLL')
    axes[1].set(xlabel='Optimizer update', ylabel='Mean negative log likelihood')
    axes[1].legend(fontsize=8)
    for axis in axes:
        axis.grid(alpha=.2)
    figure.suptitle('Mistral-7B: full dense continuation under the fixed configured bounds', fontsize=11)
    for extension in ['png', 'pdf']:
        figure.savefig(path / f'training_curve.{extension}', dpi=180)
    plt.close(figure)
    print('\n'.join(lines[:8]))


if __name__ == '__main__':
    main()
