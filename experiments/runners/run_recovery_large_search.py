"""Queue memory-checked large-model recovery attempts after TinyStories jobs."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--device', required=True)
    parser.add_argument('--wait-for-search-models', nargs='*', default=[])
    parser.add_argument('--search-output', type=Path, default=Path('results/reviewer/recovery_seed_search'))
    parser.add_argument('--max-seeds', type=int, default=1)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())

    def finished(name):
        runs = [json.loads(p.read_text()) for p in args.search_output.glob(f'{name}_bounded_seed*/result.json')
                if (p.parent / 'manifest.json').exists()]
        return len(runs) >= 5 or any(r['meets_historical_threshold'] for r in runs)

    while not (args.assets / 'assets.json').exists() or not all(finished(n) for n in args.wait_for_search_models):
        time.sleep(10)
    args.output.mkdir(parents=True, exist_ok=True)
    logs = args.output / 'logs'
    logs.mkdir(exist_ok=True)
    prefix = [sys.executable, 'experiments/runners/reviewer_recovery_large.py', '--config', str(args.config),
              '--assets', str(args.assets), '--model', args.model, '--device', args.device]
    smoke = Path('/tmp') / f'qft-recovery-{args.model}-{args.config.stem}-smoke'
    if not (smoke / 'manifest.json').exists():
        with (logs / f'{args.model}_smoke.log').open('w') as log:
            subprocess.run(prefix + ['--seed', '101', '--output', str(smoke), '--smoke', '--steps', '2', '--eval-every', '2'],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
    for seed in config['seeds'][:args.max_seeds]:
        name = f'{args.model}_bounded_seed{seed}'
        output = args.output / name
        if not (output / 'manifest.json').exists():
            if output.exists():
                raise RuntimeError(f'Incomplete attempt preserved: {output}')
            print(f'Starting {name}', flush=True)
            with (logs / f'{name}.log').open('w') as log:
                subprocess.run(prefix + ['--seed', str(seed), '--output', str(output)],
                               stdout=log, stderr=subprocess.STDOUT, check=True)
        result = json.loads((output / 'result.json').read_text())
        print(f"Completed {name}: {result['evaluation']}; qualifies={result['meets_historical_threshold']}", flush=True)
        if result['meets_historical_threshold']:
            break


if __name__ == '__main__':
    main()
