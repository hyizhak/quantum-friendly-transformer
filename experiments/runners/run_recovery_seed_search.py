"""Search a recorded seed list, stopping each model after a qualifying example."""

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
    parser.add_argument('--models', nargs='+', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--wait-for', type=Path)
    args = parser.parse_args()
    if args.wait_for:
        while not args.wait_for.exists():
            time.sleep(10)
    while not (args.assets / 'assets.json').exists():
        time.sleep(10)
    config = json.loads(args.config.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    logs = args.output / 'logs'
    logs.mkdir(exist_ok=True)
    for model in args.models:
        for seed in config['seeds']:
            name = f'{model}_bounded_seed{seed}'
            output = args.output / name
            if not (output / 'manifest.json').exists():
                if output.exists():
                    raise RuntimeError(f'Preserved incomplete attempt requires inspection: {output}')
                command = [sys.executable, str(Path(__file__).with_name('recovery_search.py')),
                           '--config', str(args.config), '--assets', str(args.assets),
                           '--model', model, '--seed', str(seed), '--device', args.device,
                           '--output', str(output)]
                print(f'Starting {name} on {args.device}', flush=True)
                with (logs / f'{name}.log').open('w') as log:
                    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            result = json.loads((output / 'result.json').read_text())
            print(f"Completed {name}: correct={result['evaluation']['correct']}/872; "
                  f"qualifies={result['meets_historical_threshold']}", flush=True)
            if result['meets_historical_threshold']:
                break


if __name__ == '__main__':
    main()
