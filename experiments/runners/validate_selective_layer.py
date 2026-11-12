"""Independently verify complete outputs and query-accounting artifacts."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def main(args):
    root = args.results
    metadata = json.loads((root / 'manifest.json').read_text())
    for path, expected in metadata['source_sha256'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected
    rows = list(csv.DictReader((root / 'layer_results.csv').open()))
    maximum_error = 0.
    for row in rows:
        n, marked, eps = int(row['N']), int(row['marked']), float(row['epsilon_ln'])
        y = 1/(2*n-1) if marked else -1.
        state = np.array([1/math.sqrt(2), -1/math.sqrt(2), y, 0.])
        sigmas = []
        for _ in range(2):
            state -= state.mean()
            sigmas.append(np.linalg.norm(state))
            state /= math.sqrt(float(state@state)+4*eps)
        saved = np.array([float(row[f'output_{i}']) for i in range(4)])
        error = float(np.linalg.norm(state-saved))
        assert error < 1e-10
        maximum_error = max(error, maximum_error)
        assert abs(float(row['sigma1'])-sigmas[0]) < 1e-10
        assert abs(float(row['sigma2'])-sigmas[1]) < 1e-10
        assert float(row['measured_input_frobenius']) <= math.sqrt(n)*(1+1e-12)
        assert float(row['maximum_weight_frobenius']) <= math.sqrt(2*math.log(n))*(1+1e-12)
        assert float(row['full_output_gap']) > .69
    queries = list(csv.DictReader((root / 'query_counts.csv').open()))
    maximum_search_error = 0.
    for row in queries:
        m = int(row['candidate_rows'])
        count = int(row['grover_iterations'])
        theta = math.asin(1/math.sqrt(m))
        rotation = np.array([[math.cos(2*theta), math.sin(2*theta)],
                             [-math.sin(2*theta), math.cos(2*theta)]])
        amplitudes = np.array([math.sin(theta), math.cos(theta)])
        for _ in range(count):
            amplitudes = rotation@amplitudes
        error = abs(float(amplitudes[0]**2)-float(row['quantum_success_given_one_mark']))
        assert error < 1e-10
        maximum_search_error = max(error, maximum_search_error)
        assert int(row['quantum_queries_including_verification']) == count+1
        classical = int(row['classical_queries_for_2over3_no_false_positives'])
        assert classical/m >= 2/3 and (classical-1)/m < 2/3
    trials = list(csv.DictReader((root / 'simulated_trials.csv').open()))
    assert len(trials) == len(queries)*len(metadata['config']['seeds'])*2*5
    assert all(0 <= int(r['success_count']) <= int(r['trials']) for r in trials)
    assert all(int(r['success_count']) == int(r['trials']) for r in trials if r['marked'] == '0')
    matched = root/'matched_search'
    matched_metadata = json.loads((matched/'manifest.json').read_text())
    assert matched_metadata['parent_manifest_sha256'] == hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest()
    for path, expected in matched_metadata['source_sha256'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected
    matched_rows = list(csv.DictReader((matched/'query_counts.csv').open()))
    for row in matched_rows:
        m, q = int(row['candidate_rows']), int(row['classical_queries_for_target'])
        p = q/m
        g = float(row['classical_guess_probability_after_miss'])
        zero_success, one_success = 1-g, p+(1-p)*g
        assert abs(zero_success-one_success) < 1e-14
        assert zero_success >= 2/3 and 1/(2-(q-1)/m) < 2/3
        count = int(row['grover_iterations'])
        theta = math.asin(1/math.sqrt(m))
        rotation = np.array([[math.cos(2*theta), math.sin(2*theta)],
                             [-math.sin(2*theta), math.cos(2*theta)]])
        amplitudes = np.array([math.sin(theta), math.cos(theta)])
        for _ in range(count):
            amplitudes = rotation@amplitudes
        assert abs(float(amplitudes[0]**2)-float(row['quantum_success_one_mark'])) < 1e-10
        assert amplitudes[0]**2 >= 2/3-1e-12
        assert int(row['quantum_queries_including_verification']) == count+1
    payload = {'source_hashes_match': True, 'complete_layer_cases_checked': len(rows),
        'query_budgets_checked': len(queries), 'simulated_trial_groups': len(trials),
        'maximum_independent_full_output_error': maximum_error,
        'maximum_independent_search_probability_error': maximum_search_error,
        'matched_success_query_budgets_checked': len(matched_rows),
        'classical_false_positives_allowed_in_matched_comparison': True,
        'all_checks_passed': True,
        'scope': 'Independent direct LayerNorm calculation and repeated two-dimensional search rotations; no quantum hardware claim.'}
    (root / 'artifact_validation.json').write_text(json.dumps(payload, indent=2)+'\n')
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path('results/reviewer/selective_layer'))
    main(parser.parse_args())

