"""Adapt capped Mistral weights only if head/norm/cap tuning still fails."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prior-run', type=Path, required=True)
    parser.add_argument('--initial-run', type=Path, required=True)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', required=True)
    parser.add_argument('--seed', type=int, default=101)
    args = parser.parse_args()
    prior = args.prior_run.resolve()
    while not (prior / 'manifest.json').exists():
        initial = args.initial_run.resolve()
        if (initial / 'manifest.json').exists() and json.loads((initial / 'result.json').read_text())['meets_historical_threshold']:
            print('Mistral already recovered; no low-rank run.', flush=True)
            return
        time.sleep(10)
    if json.loads((prior / 'result.json').read_text())['meets_historical_threshold']:
        print('Mistral already recovered; no low-rank run.', flush=True)
        return
    root = args.output.resolve()
    (root / 'logs').mkdir(parents=True, exist_ok=True)
    prefix = [sys.executable, str(Path(__file__).with_name('recovery_low_rank.py')),
              '--config', str(args.config.resolve()),
              '--assets', str(args.assets.resolve()), '--seed', str(args.seed), '--device', args.device]
    smoke = root / 'smoke'
    if not (smoke / 'manifest.json').exists():
        with (root / 'logs/smoke.log').open('w') as log:
            subprocess.run(prefix + ['--output', str(smoke), '--smoke', '--steps', '2', '--eval-every', '2'],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
    output = root / f'mistral7b_bounded_seed{args.seed}'
    print('Starting capped low-rank Mistral recovery.', flush=True)
    with (root / f'logs/mistral7b_bounded_seed{args.seed}.log').open('w') as log:
        subprocess.run(prefix + ['--output', str(output)], stdout=log, stderr=subprocess.STDOUT, check=True)
    result = json.loads((output / 'result.json').read_text())
    print(f"Completed low-rank Mistral: {result['evaluation']}; qualifies={result['meets_historical_threshold']}", flush=True)


if __name__ == '__main__':
    main()
