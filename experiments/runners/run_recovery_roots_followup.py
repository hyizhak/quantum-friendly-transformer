"""Try original training sentences only if a model remains unrecovered."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['2layers33m', 'olmo1b'], required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--device', required=True)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prior-root', type=Path, required=True)
    args = parser.parse_args()
    root = args.results.resolve()
    prior = args.prior_root.resolve()
    required = 3 if args.model == '2layers33m' else 1
    def recovered():
        for p in root.glob(f'recovery*/*{args.model}_bounded_seed*/result.json'):
            if (p.parent / 'manifest.json').exists():
                result = json.loads(p.read_text())
                if result['name'] == args.model and result['meets_historical_threshold']:
                    return True
        return False
    while len(list(prior.glob(f'{args.model}_bounded_seed*/manifest.json'))) < required:
        if recovered():
            print(f'{args.model} already recovered; no root-only run.', flush=True)
            return
        time.sleep(10)
    config = json.loads(args.config.read_text())
    out = args.output.resolve()
    (out / 'logs').mkdir(parents=True, exist_ok=True)
    for seed in config['seeds']:
        if recovered():
            print(f'{args.model} already recovered; stopping.', flush=True)
            return
        name = f'{args.model}_bounded_seed{seed}'
        command = [sys.executable, str(Path(__file__).with_name('recovery_roots.py')), '--model', args.model,
                   '--config', str(args.config), '--assets', str(args.assets.resolve()),
                   '--device', args.device, '--seed', str(seed), '--output', str(out / name)]
        print(f'Starting {name} on matched original training sentences.', flush=True)
        with (out / 'logs' / f'{name}.log').open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        result = json.loads((out / name / 'result.json').read_text())
        print(f"Completed {name}: {result['evaluation']}; qualifies={result['meets_historical_threshold']}", flush=True)


if __name__ == '__main__':
    main()
