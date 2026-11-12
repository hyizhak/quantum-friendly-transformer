"""Run follow-up adaptation only if a missing model's initial attempt fails."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--device', required=True)
    args = parser.parse_args()
    prior = Path('results/reviewer/recovery_large_search') / f'{args.model}_bounded_seed101'
    while not (prior / 'manifest.json').exists():
        time.sleep(10)
    result = json.loads((prior / 'result.json').read_text())
    if result['meets_historical_threshold']:
        print(f'{args.model} recovered; no follow-up run.', flush=True)
        return
    command = [sys.executable, 'experiments/runners/run_recovery_large_search.py',
               '--model', args.model, '--config', str(args.config), '--device', args.device,
               '--assets', '/tmp/qft-recovery-search-assets',
               '--output', 'results/reviewer/recovery_large_adaptation']
    print(f'{args.model} still missing; starting follow-up adaptation.', flush=True)
    subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
