"""Adapt capped Mistral weights only if head/norm/cap tuning still fails."""

import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    prior = Path('results/reviewer/recovery_large_adaptation/mistral7b_bounded_seed101')
    while not (prior / 'manifest.json').exists():
        initial = Path('results/reviewer/recovery_large_search/mistral7b_bounded_seed101')
        if (initial / 'manifest.json').exists() and json.loads((initial / 'result.json').read_text())['meets_historical_threshold']:
            print('Mistral already recovered; no low-rank run.', flush=True)
            return
        time.sleep(10)
    if json.loads((prior / 'result.json').read_text())['meets_historical_threshold']:
        print('Mistral already recovered; no low-rank run.', flush=True)
        return
    root = Path('results/reviewer/recovery_low_rank')
    (root / 'logs').mkdir(parents=True, exist_ok=True)
    prefix = [sys.executable, 'experiments/runners/reviewer_recovery_low_rank.py',
              '--config', 'experiments/configs/reviewer/recovery_mistral7b_low_rank.json',
              '--assets', '/tmp/qft-recovery-search-assets', '--seed', '101', '--device', 'cuda:0']
    smoke = Path('/tmp/qft-recovery-mistral-low-rank-smoke')
    if not (smoke / 'manifest.json').exists():
        with (root / 'logs/smoke.log').open('w') as log:
            subprocess.run(prefix + ['--output', str(smoke), '--smoke', '--steps', '2', '--eval-every', '2'],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
    output = root / 'mistral7b_bounded_seed101'
    print('Starting capped low-rank Mistral recovery.', flush=True)
    with (root / 'logs/mistral7b_bounded_seed101.log').open('w') as log:
        subprocess.run(prefix + ['--output', str(output)], stdout=log, stderr=subprocess.STDOUT, check=True)
    result = json.loads((output / 'result.json').read_text())
    print(f"Completed low-rank Mistral: {result['evaluation']}; qualifies={result['meets_historical_threshold']}", flush=True)


if __name__ == '__main__':
    main()
