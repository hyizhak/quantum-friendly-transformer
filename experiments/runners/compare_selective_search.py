"""Compare search budgets with a common worst-case success target on both promises."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

import numpy as np

from experiments.lib.reviewer import manifest, save_csv, save_json


def main(args):
    root = args.results
    config = json.loads((root/'config.json').read_text())
    output = root/'matched_search'
    output.mkdir(exist_ok=False)
    target = 2/3
    rows, trials = [], []
    for n in config['lengths']:
        m = n-1
        theta = math.asin(1/math.sqrt(m))
        rotations = max(0,math.ceil((math.asin(math.sqrt(target))/theta-1)/2))
        q = rotations+1
        quantum_success = math.sin((2*rotations+1)*theta)**2
        assert quantum_success >= target
        assert rotations == 0 or math.sin((2*rotations-1)*theta)**2 < target
        classical_q = math.ceil(m/2)
        # Sample q distinct positions, then guess "one mark" with probability
        # g=(1-p)/(2-p) if none were found. Both promise cases have success
        # 1/(2-p). This is a stronger baseline than forbidding false positives.
        equal_p = min(1.,q/m)
        equal_g = (1-equal_p)/(2-equal_p)
        classical_p = classical_q/m
        classical_g = (1-classical_p)/(2-classical_p)
        assert 1-classical_g >= target
        rows.append({'N':n,'candidate_rows':m,'success_target':target,
            'grover_iterations':rotations,'quantum_queries_including_verification':q,
            'quantum_success_zero_marks':1.,'quantum_success_one_mark':quantum_success,
            'classical_queries_for_target':classical_q,
            'classical_worst_case_success_at_target_budget':1-classical_g,
            'classical_guess_probability_after_miss':classical_g,
            'classical_worst_case_success_at_equal_quantum_budget':1-equal_g,
            'classical_to_quantum_query_ratio':classical_q/q})
        for seed in config['seeds']:
            rng=np.random.default_rng(seed+n+941)
            count=config['trials_per_seed']
            for marked in [0,1]:
                for method,budget,p,g in [
                    ('classical_balanced_equal_queries',q,equal_p,equal_g),
                    ('classical_balanced_target_budget',classical_q,classical_p,classical_g)]:
                    # Exact distribution of a uniformly located unique mark
                    # in a uniformly chosen distinct subset, followed by a coin.
                    found=rng.binomial(1,p if marked else 0.,count).astype(bool)
                    guessed=rng.binomial(1,g,count).astype(bool)
                    predicted=found|(~found&guessed)
                    success=int((predicted==bool(marked)).sum())
                    trials.append({'N':n,'seed':seed,'marked':marked,'method':method,
                        'queries':budget,'trials':count,'success_count':success})
                found=rng.binomial(1,quantum_success if marked else 0.,count)
                trials.append({'N':n,'seed':seed,'marked':marked,'method':'simulated_grover_target_budget',
                    'queries':q,'trials':count,'success_count':int((found==marked).sum())})
    save_csv(output/'query_counts.csv',rows)
    save_csv(output/'simulated_trials.csv',trials)
    save_json(output/'manifest.json',manifest(config,[Path(__file__),ROOT/'experiments/lib/reviewer.py'],
        parent_manifest_sha256=hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest(),
        success_target=target,output_error=config['output_error'],
        scope='Ideal query-model comparison with errors allowed in both classical promise cases; not hardware runtime or a new proof of a lower bound.'))
    save_json(output/'summary.json',{'largest_case':rows[-1],'lengths':len(rows),
        'minimum_quantum_worst_case_success':min(r['quantum_success_one_mark'] for r in rows),
        'minimum_classical_worst_case_success_at_target_budget':min(r['classical_worst_case_success_at_target_budget'] for r in rows),
        'simulation_trials':sum(r['trials'] for r in trials)})
    print(json.dumps(rows[-1],indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,default=Path('results/reviewer/selective_layer'))
    main(parser.parse_args())
