"""Numerical complete-layer check and simulated query accounting for selectivity."""

import argparse
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

import numpy as np
import torch

from experiments.lib import selective_layer as family
from experiments.lib.experiment import manifest, save_csv, save_json


@torch.no_grad()
def main(args):
    config = json.loads(args.config.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    save_json(args.output / 'config.json', config)
    torch.set_num_threads(config['threads'])
    started = time.monotonic()
    layers, query_rows, trials, dense_checks = [], [], [], []
    for n in config['lengths']:
        absent, present = family.input_rows(n), family.input_rows(n, n//2)
        assert torch.equal(absent.abs(), present.abs())
        alpha_s = math.sqrt(n)
        assert float(present.norm()) <= alpha_s*(1+1e-12)
        alpha_w = math.sqrt(2*math.log(n))
        original_outputs = None
        for eps in config['layer_norm_epsilons']:
            model = family.SelectiveLayer(n, eps=eps)
            out0, d0 = model(absent)
            out1, d1 = model(present)
            expected0, expected1 = model.from_average(-1.), model.from_average(1/(2*n-1))
            discrepancy = max(float((out0-expected0).norm()), float((out1-expected1).norm()))
            assert discrepancy < 1e-10
            gap = float((out1-out0).norm())
            assert gap > .69
            maximum_norm = max(float(p.norm()) for p in model.parameters())
            assert maximum_norm <= alpha_w*(1+1e-12)
            for marked, output, diag in [(0, out0, d0), (1, out1, d1)]:
                layers.append({'N': n, 'marked': marked, 'epsilon_ln': eps,
                    'attention_denominator': model.denominator, 'alpha_s_frobenius_bound': alpha_s,
                    'measured_input_frobenius': float(present.norm()),
                    'alpha_w': alpha_w, 'maximum_weight_frobenius': maximum_norm,
                    'q_frobenius': float(model.q.weight.norm()), 'k_frobenius': float(model.k.weight.norm()),
                    'v_frobenius': float(model.v.weight.norm()), 'ffn_frobenius': 0.,
                    'kappa_encoding_bound': alpha_s**2*alpha_w**2/model.denominator,
                    'logit_range': float(diag['logits'].max()-diag['logits'].min()),
                    'maximum_probability': float(diag['probabilities'].max()),
                    'attention_average': float(diag['attention_average']),
                    'sigma1': float(diag['sigma1']), 'sigma2': float(diag['sigma2']),
                    'output_norm': float(output.norm()), 'full_output_gap': gap,
                    'canonical_output_error': discrepancy,
                    **{f'output_{i}': float(output[i]) for i in range(4)}})
            if eps == 0:
                original_outputs = (model, out0, out1)
        assert original_outputs is not None
        model, out0, out1 = original_outputs
        candidates = n-1
        rotations, quantum_probability = family.grover_parameters(candidates)
        queries = rotations+1
        classical_queries = math.ceil((2/3)*candidates)
        assert quantum_probability > 2/3
        if candidates <= config['dense_statevector_limit']:
            for marked in [None, 0, candidates-1]:
                state = family.simulate_search_state(candidates, marked, rotations)
                norm_error = abs(float(np.square(state).sum())-1.)
                probability_error = abs(float(state[marked]**2)-quantum_probability) if marked is not None else float(np.max(np.abs(state-1/math.sqrt(candidates))))
                assert norm_error < 1e-10 and probability_error < 1e-10
                dense_checks.append({'N': n, 'marked_index': -1 if marked is None else marked,
                                     'state_norm_error': norm_error, 'probability_error': probability_error})
        query_rows.append({'N': n, 'candidate_rows': candidates, 'grover_iterations': rotations,
            'quantum_queries_including_verification': queries, 'quantum_success_given_one_mark': quantum_probability,
            'quantum_success_given_zero_marks': 1.,
            'classical_equal_query_success_given_one_mark': min(1., queries/candidates),
            'classical_queries_for_2over3_no_false_positives': classical_queries,
            'classical_to_quantum_query_ratio': classical_queries/queries,
            'full_output_error_tolerance': config['output_error'],
            'sqrt_candidates': math.sqrt(candidates)})
        for seed in config['seeds']:
            rng = np.random.default_rng(seed+n)
            count = config['trials_per_seed']
            for marked in [0, 1]:
                target = out1 if marked else out0
                for method, budget, probability in [
                    ('simulated_grover_with_verification', queries, quantum_probability if marked else 0.),
                    ('classical_distinct_equal_queries', queries, min(1.,queries/candidates) if marked else 0.),
                    ('classical_distinct_2over3_budget', classical_queries, classical_queries/candidates if marked else 0.)]:
                    found = rng.binomial(1, probability, count)
                    outputs = torch.where(torch.tensor(found, dtype=torch.bool)[:, None], out1, out0)
                    errors = (outputs-target).norm(dim=-1).numpy()
                    trials.append({'N': n, 'seed': seed, 'marked': marked, 'method': method,
                        'queries': budget, 'trials': count, 'success_count': int((errors <= config['output_error']).sum()),
                        'mean_output_error': float(errors.mean()), 'maximum_output_error': float(errors.max())})
                for method, budget in [('self_normalized_equal_queries',queries), ('self_normalized_4000',config['sampler_budget'])]:
                    hits = rng.binomial(budget, 1/n, count) if marked else np.zeros(count, dtype=np.int64)
                    average = (hits*n-(budget-hits))/(hits*n+(budget-hits))
                    outputs = model.from_average(average)
                    errors = (outputs-target).norm(dim=-1).numpy()
                    trials.append({'N': n, 'seed': seed, 'marked': marked, 'method': method,
                        'queries': budget, 'trials': count, 'success_count': int((errors <= config['output_error']).sum()),
                        'mean_output_error': float(errors.mean()), 'maximum_output_error': float(errors.max())})
        print(f'N={n}: gap={float((out1-out0).norm()):.6f}, verified quantum queries={queries}, '
              f'one-mark success={quantum_probability:.6f}, classical 2/3 budget={classical_queries}', flush=True)
    save_csv(args.output / 'layer_results.csv', layers)
    save_csv(args.output / 'query_counts.csv', query_rows)
    save_csv(args.output / 'simulated_trials.csv', trials)
    save_csv(args.output / 'dense_statevector_checks.csv', dense_checks)
    sources = [Path(__file__), Path(family.__file__), ROOT / 'experiments/lib/experiment.py',
               ROOT / 'src/quantum_friendly_transformer/norm_transformer/normalization.py']
    save_json(args.output / 'manifest.json', manifest(config, sources, seconds=time.monotonic()-started,
        dtype='float64', purpose='Complete-layer realization of the explicit selective family in response.tex',
        interpretation='A numerical construction and classical simulation of a query model; not quantum hardware performance.',
        limitations=['No learned model or pretrained-performance claim.',
                     'Growing Q/K norms and native denominator place this outside the diffuse fixed-floor regime.',
                     'Large-N Grover probabilities use the exact two-dimensional rotation formula; small cases also use dense state vectors.',
                     'Classical query budget is for the stated no-false-positive search algorithm, not an exact universal optimal constant.',
                     'The formal general lower bound and the manuscript proof require separate mathematical verification.',
                     'Oracle counts omit gate synthesis, data loading, noise, and hardware timing.'],
        sources=['https://arxiv.org/abs/quant-ph/9605043']))
    summary = {'tested_lengths': config['lengths'], 'minimum_full_output_gap': min(r['full_output_gap'] for r in layers),
               'minimum_sigma1': min(r['sigma1'] for r in layers), 'minimum_sigma2': min(r['sigma2'] for r in layers),
               'maximum_canonical_output_error': max(r['canonical_output_error'] for r in layers),
               'minimum_quantum_success_given_one_mark': min(r['quantum_success_given_one_mark'] for r in query_rows),
               'largest_case': query_rows[-1], 'dense_statevector_cases': len(dense_checks),
               'simulation_trials': sum(r['trials'] for r in trials)}
    save_json(args.output / 'summary.json', summary)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'experiments/configs/selective_attention/selective_layer.json')
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())

