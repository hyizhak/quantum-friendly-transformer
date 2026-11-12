"""Try archived selection policy only for a model still missing after prior searches."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['33m', '2layers33m'], required=True)
    parser.add_argument('--device', required=True)
    parser.add_argument('--prior-root', type=Path)
    parser.add_argument('--prior-runs', type=int, default=3)
    parser.add_argument('--config', type=Path, default=Path('experiments/configs/reviewer/recovery_matched_protocol.json'))
    parser.add_argument('--runner', default='experiments/runners/reviewer_recovery_matched.py')
    parser.add_argument('--output-root', type=Path, default=Path('results/reviewer/recovery_matched_protocol'))
    args = parser.parse_args()
    root = Path('results/reviewer')
    previous = args.prior_root or root / ('recovery_longer_search' if args.model == '33m' else 'recovery_gentler_search')
    def already_recovered():
        for p in root.glob(f'recovery*/*{args.model}_bounded_seed*/result.json'):
            if not (p.parent / 'manifest.json').exists():
                continue
            result = json.loads(p.read_text())
            if result['name'] == args.model and result['meets_historical_threshold']:
                return True
        return False
    while len(list(previous.glob(f'{args.model}_bounded_seed*/manifest.json'))) < args.prior_runs:
        if already_recovered():
            print(f'{args.model} already recovered; no matched-policy run.', flush=True)
            return
        time.sleep(10)
    config_path = args.config
    config = json.loads(config_path.read_text())
    output_root = args.output_root
    (output_root / 'logs').mkdir(parents=True, exist_ok=True)
    for seed in config['seeds']:
        if already_recovered():
            print(f'{args.model} recovered; stopping.', flush=True)
            return
        name = f'{args.model}_bounded_seed{seed}'
        output = output_root / name
        command = [sys.executable, args.runner,
                   '--config', str(config_path), '--assets', '/tmp/qft-recovery-search-assets',
                   '--output', str(output), '--model', args.model, '--seed', str(seed), '--device', args.device]
        print(f'Starting {name} with full training / validation selection.', flush=True)
        with (output_root / 'logs' / f'{name}.log').open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        result = json.loads((output / 'result.json').read_text())
        print(f"Completed {name}: {result['evaluation']}; qualifies={result['meets_historical_threshold']}", flush=True)


if __name__ == '__main__':
    main()
