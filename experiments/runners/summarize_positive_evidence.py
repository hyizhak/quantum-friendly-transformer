"""Collect the positive follow-up outcomes with their full experimental scope."""

import argparse
import csv
import json
from pathlib import Path


def main(args):
    root = args.results
    plan = json.loads((root / 'recovery_sam/attempt_plan.json').read_text())
    rows = []
    for item in plan:
        if item['status'] != 'complete':
            continue
        path = root / f'recovery_sam/2layers33m_bounded_seed{item["seed"]}'
        result = json.loads((path / 'result.json').read_text())
        config = json.loads((path / 'config.json').read_text())
        rows.append({'seed': item['seed'], 'correct': result['evaluation']['correct'],
            'accuracy': result['evaluation']['correct']/result['evaluation']['examples'],
            'mean_nll': result['evaluation']['mean_nll'], 'selected_step': result['selected_step'],
            'optimizer_steps': result['optimizer_steps'], 'sam_radius': config['sam_radius'],
            'learning_rate': config['models']['2layers33m']['learning_rate'],
            'root_fraction': config['root_fraction'], 'qualifies': result['meets_historical_threshold'],
            'result_path': str(path / 'result.json')})
    best = max(rows, key=lambda r: (r['accuracy'], -r['mean_nll']))
    story = json.loads((root / 'recovery_story.json').read_text())
    coverage = json.loads((root / 'recovery_validation_coverage.json').read_text())
    selective = json.loads((root / 'selective_layer/summary.json').read_text())
    selective_audit = json.loads((root / 'selective_layer/artifact_validation.json').read_text())
    matched_search = json.loads((root / 'selective_layer/matched_search/summary.json').read_text())
    result = json.loads(Path(best['result_path']).read_text())
    parent = json.loads(Path(result['parent_checkpoint']['result_path']).read_text())
    reports = [r for p in (root / 'recovery_sam').glob('checkpoint_replay*.json')
               for r in json.loads(p.read_text()).get('reports', [])]
    replay = next((r for r in reports if r['run']==Path(best['result_path']).parent.name), None)
    payload = {'distinct_recovered_models': story['distinct_models_recovered'], 'total_models': story['total_models'],
        'sam_attempts': rows, 'best_sam': best, 'best_sam_replay': replay,
        'skipped_after_recovery': [r['seed'] for r in plan if r['status']=='skipped_after_recovery'],
        'selective_family': selective, 'selective_validation': selective_audit, 'matched_search': matched_search,
        'completed_bounded_attempts': coverage['completed_bounded_attempts'],
        'artifact_verified_attempts': coverage['artifact_verified_attempts'],
        'fresh_prediction_replays': coverage['fresh_prediction_replays'],
        'reporting_scope': 'Selected recovery examples and a constructed query-model simulation; no independent SST-2 test or quantum hardware timing.'}
    (root / 'positive_evidence_summary.json').write_text(json.dumps(payload,indent=2)+'\n')
    with (root / 'recovery_sam_attempts.csv').open('w',newline='') as handle:
        writer = csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    last = matched_search['largest_case']
    lines = ['POSITIVE EVIDENCE FOLLOW-UP', '',
        f"Distinct original models recovered under configured floors: {story['distinct_models_recovered']}/{story['total_models']}.",
        f"TinyStories-2Layers-33M: {parent['evaluation']['correct']}/872 -> {best['correct']}/872 ({100*best['accuracy']:.4f}%).",
        f"Original threshold: {100*result['historical_recovery_threshold']:.4f}% (740/872 required); qualifies={best['qualifies']}.",
        f"NLL: {parent['evaluation']['mean_nll']:.6f} -> {best['mean_nll']:.6f}.",
        f"Selected seed {best['seed']}, update {best['selected_step']}, SAM radius {best['sam_radius']}, peak LR {best['learning_rate']}.",
        'Full parameters trained; gold labels only; half original training roots and half original training phrases.',
        'Same cap 24, alpha_s=32, and denominator floor 589824 throughout both SAM passes.', '',
        'Completed attempts (all disclosed): ' + ', '.join(f"seed {r['seed']}: {r['correct']}/872" for r in rows) + '.',
        'Remaining planned attempts skipped after recovery: ' + ', '.join(map(str,payload['skipped_after_recovery'])) + '.',
        (f"Fresh selected-checkpoint replay: predictions match={replay['all_predictions_match']}; maximum per-example NLL difference={replay['maximum_per_example_nll_difference']}." if replay else 'Fresh selected-checkpoint replay: pending.'),
        'The six previously recovered models were not retrained. OLMo and Mistral remain unrecovered.', '',
        'Complete-layer selective example:',
        f"For N=16 through {last['N']:,}, the full output gap stays >= {selective['minimum_full_output_gap']:.6f}.",
        f"Both normalizations remain conditioned: minimum centered norms {selective['minimum_sigma1']:.6f}, {selective['minimum_sigma2']:.6f}.",
        'The marked token receives over half the attention probability. Unit input rows give an exact Frobenius input bound.',
        f"At N={last['N']:,}, ideal Grover search plus verification uses {last['quantum_queries_including_verification']} queries with calculated one-mark success {last['quantum_success_one_mark']:.9f} (zero-mark success 1).",
        f"The implemented balanced classical search, allowing false positives, uses {last['classical_queries_for_target']:,} queries for success >=2/3 in both promise cases.",
        'This is a constructed selective regime with growing Q/K norms and native denominator 2.',
        'It is a classical simulation of ideal query access, not quantum hardware execution or the general polynomial pipeline.',
        'The independent validator checks all 36 complete-layer cases and all nine query budgets.', '',
        'Existing positive replication evidence:',
        'The fixed lower-LR TinyStories-8M cohort recovered in all three seeds: 83.30 +/- 0.46% accuracy.',
        'Its matched uniform-control mean was 82.87 +/- 0.89%; no statistically established superiority is claimed.',
        'See recovery_extension_comparison.txt for every setting and incomplete control.', '',
        'Limits for the paper:',
        'Recovery is validation-selected and compared with historical native targets; it is not an unbiased seed average or independent test.',
        'The configured pretrained alpha_s values do not certify complete input-matrix bounds or the theorem architecture.',
        'The selective-family construction supplies numerical support; the formal lower bound remains a mathematical claim.',
        f"Cumulative audit coverage: {coverage['artifact_verified_attempts']}/{coverage['completed_bounded_attempts']} completed new bounded attempts; {coverage['fresh_prediction_replays']} fresh replays.", '',
        'Artifacts: recovery_sam_attempts.csv, recovery_story_table.tex,',
        'figures/recovery_sam.pdf, selective_layer/selective_layer.pdf, selective_layer/construction.txt.',
        'Method references: https://arxiv.org/abs/2010.01412 and https://arxiv.org/abs/quant-ph/9605043']
    (root / 'positive_evidence_summary.txt').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(7.5,4),constrained_layout=True)
    for row in rows:
        selection = list(csv.DictReader(Path(row['result_path']).with_name('selection.csv').open()))
        axis.plot([int(r['step']) for r in selection], [100*int(r['correct'])/int(r['examples']) for r in selection],
                  label=f"Seed {row['seed']}, rho={row['sam_radius']}")
    axis.axhline(100*result['historical_recovery_threshold'],color='#922637',linestyle='--',label='Original recovery threshold')
    axis.axhline(100*parent['evaluation']['correct']/872,color='#777777',linestyle=':',label='Previous best')
    axis.scatter([best['selected_step']],[100*best['accuracy']],color='#922637',zorder=5)
    axis.set(title='Two-layer model: sharpness-aware bounded continuation',xlabel='Optimizer update',ylabel='Validation accuracy (%)')
    axis.legend(fontsize=8)
    axis.grid(alpha=.2)
    for extension in ['png','pdf']:
        figure.savefig(root / f'figures/recovery_sam.{extension}',dpi=180)
    plt.close(figure)
    print('\n'.join(lines[:14]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,default=Path('results/reviewer'))
    main(parser.parse_args())
