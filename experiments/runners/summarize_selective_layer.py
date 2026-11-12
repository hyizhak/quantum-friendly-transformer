"""Create the selective-family result table, report, and exportable figure."""

import argparse
import csv
import json
from pathlib import Path


def main(args):
    root = args.results
    summary = json.loads((root / 'summary.json').read_text())
    queries = list(csv.DictReader((root / 'query_counts.csv').open()))
    trials = list(csv.DictReader((root / 'simulated_trials.csv').open()))
    layers = list(csv.DictReader((root / 'layer_results.csv').open()))
    last = summary['largest_case']
    matched_summary = json.loads((root / 'matched_search/summary.json').read_text())
    matched_last = matched_summary['largest_case']
    lines = ['COMPLETE-LAYER SELECTIVE-ATTENTION EXAMPLE', '',
        'Constructed dimension-four family with unit input rows, native attention denominator 2,',
        'Q/K Frobenius norms sqrt(2 log N), value norm sqrt(2), zero FFN, and two residual/LayerNorm operations.',
        'This is a different regime from the fixed-floor recovery table.', '',
        f"Nine lengths from 16 to {last['N']:,}; both zero-mark and one-mark cases; LN epsilon 0 and 1e-5.",
        f"Minimum complete-output Euclidean separation: {summary['minimum_full_output_gap']:.6f}.",
        f"Minimum measured centered norms: sigma1={summary['minimum_sigma1']:.6f}, sigma2={summary['minimum_sigma2']:.6f}.",
        f"Maximum discrepancy between dense attention and the known family output: {summary['maximum_canonical_output_error']:.3g}.",
        'A present marked token receives probability N/(2N-1), exceeding 1/2 at every tested length.', '',
        'Query-model illustration at full-output error tolerance 0.1:',
        f"At N={last['N']:,}, Grover search plus final candidate verification uses {last['quantum_queries_including_verification']:,} coherent/value queries.",
        f"Its calculated one-mark success probability is {last['quantum_success_given_one_mark']:.9f}; zero-mark success is 1.",
        f"The implemented random search without replacement needs {last['classical_queries_for_2over3_no_false_positives']:,} queries for success >=2/3 with no false positives.",
        f"At the equal {last['quantum_queries_including_verification']}-query budget its one-mark success is {last['classical_equal_query_success_given_one_mark']:.6f}.",
        f"Dense state-vector checks cover {summary['dense_statevector_cases']} small cases; large sizes use the exact search rotation formula.",
        f"Saved Monte Carlo query-model trials: {summary['simulation_trials']:,} across five seeds.", '',
        'Stronger matched-success comparison (primary query figure):',
        'Allow the classical algorithm to guess after misses, balancing its errors between both promise cases.',
        f"At N={matched_last['N']:,}, success >=2/3 in both cases requires {matched_last['quantum_queries_including_verification']:,} queries for the simulated Grover routine",
        f"and {matched_last['classical_queries_for_target']:,} for the implemented balanced classical search.",
        f"Calculated Grover one-mark success: {matched_last['quantum_success_one_mark']:.9f}; classical worst-case success: {matched_last['classical_worst_case_success_at_target_budget']:.9f}.",
        'This comparison permits classical false positives and does not rely on the earlier no-false-positive restriction.',
        'The earlier high-success and no-false-positive comparison is retained in the original tables.', '',
        'Interpretation: positive numerical support for the complete-layer selective construction and its stated search-query mechanism.',
        'The formal lower bound and the manuscript proof remain separate mathematical claims.',
        'These are classical simulations with ideal coherent oracle access, not measured quantum execution or wall-clock speedups.',
        'No trained-model accuracy or general attention-sharpness threshold is claimed.',
        'The classical counts describe explicit algorithms; the experiment is not a universal lower-bound proof.',
        'Files: layer_results.csv, query_counts.csv, simulated_trials.csv, dense_statevector_checks.csv, artifact_validation.json.',
        'Search reference: https://arxiv.org/abs/quant-ph/9605043']
    (root / 'summary.txt').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    queries = list(csv.DictReader((root / 'matched_search/query_counts.csv').open()))
    trials = list(csv.DictReader((root / 'matched_search/simulated_trials.csv').open()))
    figure, axes = plt.subplots(1, 3, figsize=(12.5, 3.8), constrained_layout=True)
    n = [int(row['N']) for row in queries]
    marked = [row for row in layers if row['marked']=='1' and float(row['epsilon_ln'])==0]
    axes[0].semilogx(n, [float(row['full_output_gap']) for row in marked], 'o-', label='Complete-output separation')
    axes[0].semilogx(n, [float(row['maximum_probability']) for row in marked], 's-', label='Marked-key probability')
    axes[0].axhline(.5, color='#888888', linestyle=':', linewidth=1)
    axes[0].set(title='Selectivity survives the complete layer', xlabel='Rows N', ylim=(.45,.78))
    axes[0].legend(fontsize=8)
    axes[1].loglog(n, [int(row['quantum_queries_including_verification']) for row in queries], 'o-', label='Grover + verification')
    axes[1].loglog(n, [int(row['classical_queries_for_target']) for row in queries], 's-', label='Classical search with guessing')
    axes[1].set(title='Both methods: success >=2/3', xlabel='Rows N', ylabel='Oracle queries')
    axes[1].legend(fontsize=8)
    for field, label in [('quantum_success_one_mark','Grover + verification'),
                          ('classical_worst_case_success_at_equal_quantum_budget','Classical search with guessing')]:
        probabilities = [float(row[field]) for row in queries]
        axes[2].semilogx(n, probabilities, 'o-', label=label)
    axes[2].axhline(2/3,color='#888888',linestyle=':',linewidth=1)
    axes[2].set(title='Exact success probabilities, equal queries', xlabel='Rows N', ylabel='Worst-case Pr(output error <=0.1)', ylim=(.4,1.03))
    axes[2].legend(fontsize=8)
    for axis in axes:
        axis.grid(alpha=.2)
    figure.suptitle('Constructed selective-attention family; ideal query-model simulation', fontsize=12)
    for extension in ['png','pdf']:
        figure.savefig(root / f'selective_layer.{extension}', dpi=180)
    plt.close(figure)
    print('\n'.join(lines[:18]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    main(parser.parse_args())
