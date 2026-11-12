"""Continue the best bounded parent only after the pending searches fail."""

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
    required = 8 if args.model == '2layers33m' else 2

    def completed():
        rows = []
        for path in root.glob(f'recovery*/{args.model}_bounded_seed*/result.json'):
            if (path.parent / 'manifest.json').exists():
                result = json.loads(path.read_text())
                if not result['smoke']:
                    rows.append((path, result))
        return rows

    def recovered():
        return any(result['meets_historical_threshold'] for _, result in completed())

    while len(list(prior.glob(f'{args.model}_bounded_seed*/manifest.json'))) < required:
        if recovered():
            print(f'{args.model} already recovered; no continuation.', flush=True)
            return
        time.sleep(10)
    if recovered():
        print(f'{args.model} already recovered; no continuation.', flush=True)
        return
    parent_path, parent = max(completed(), key=lambda r: (r[1]['evaluation']['correct'], -r[1]['evaluation']['mean_nll']))
    config = json.loads(args.config.read_text())
    config['parent_checkpoint'] = {'path': str(parent_path.parent / 'selected.pth'),
                                   'sha256': parent['checkpoint_sha256'],
                                   'result_path': str(parent_path),
                                   'selected_step': parent['selected_step']}
    out = args.output.resolve()
    (out / 'logs').mkdir(parents=True, exist_ok=True)
    config_path = out / (args.model + '_config.json')
    with config_path.open('x') as handle:
        json.dump(config, handle, indent=2)
        handle.write('\n')
    print(f'Continuing {parent_path}: {parent["evaluation"]}', flush=True)
    for seed in config['seeds']:
        if recovered():
            print(f'{args.model} recovered; stopping.', flush=True)
            return
        name = f'{args.model}_bounded_seed{seed}'
        command = [sys.executable, str(Path(__file__).with_name('recovery_curriculum.py')),
                   '--model', args.model, '--config', str(config_path),
                   '--assets', str(args.assets.resolve()), '--seed', str(seed),
                   '--device', args.device, '--output', str(out / name)]
        with (out / 'logs' / (name + '.log')).open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        result = json.loads((out / name / 'result.json').read_text())
        print(f"Completed {name}: {result['evaluation']}; qualifies={result['meets_historical_threshold']}", flush=True)


if __name__ == '__main__':
    main()
