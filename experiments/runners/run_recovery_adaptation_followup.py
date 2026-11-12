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
    parser.add_argument('--prior-run', type=Path, required=True)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prior = args.prior_run.resolve()
    while not (prior / 'manifest.json').exists():
        time.sleep(10)
    result = json.loads((prior / 'result.json').read_text())
    if result['meets_historical_threshold']:
        print(f'{args.model} recovered; no follow-up run.', flush=True)
        return
    command = [sys.executable, str(Path(__file__).with_name('run_recovery_large_search.py')),
               '--model', args.model, '--config', str(args.config), '--device', args.device,
               '--assets', str(args.assets.resolve()),
               '--output', str(args.output.resolve())]
    print(f'{args.model} still missing; starting follow-up adaptation.', flush=True)
    subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
