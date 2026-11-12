"""Try classifier-only sentence adaptation if the pending missing-model recipes fail."""

import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['2layers33m', 'olmo1b', 'mistral7b'], required=True)
    parser.add_argument('--device', required=True)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prior-root', type=Path, required=True)
    args = parser.parse_args()
    root = args.results.resolve()
    prior = args.prior_root.resolve()
    required = 3 if args.model == '2layers33m' else 1

    def completed():
        found = []
        for path in root.glob(f'recovery*/{args.model}_bounded_seed*/result.json'):
            if (path.parent / 'manifest.json').exists():
                result = json.loads(path.read_text())
                if not result['smoke']:
                    found.append((path, result))
        return found

    while len(list(prior.glob(f'{args.model}_bounded_seed*/manifest.json'))) < required:
        if any(result['meets_historical_threshold'] for _, result in completed()):
            print(f'{args.model} recovered; no classifier-only run.', flush=True)
            return
        time.sleep(10)
    rows = completed()
    if any(result['meets_historical_threshold'] for _, result in rows):
        print(f'{args.model} recovered; no classifier-only run.', flush=True)
        return
    path, result = max(rows, key=lambda r: (r[1]['evaluation']['correct'], -r[1]['evaluation']['mean_nll']))
    config = copy.deepcopy(json.loads((path.parent / 'config.json').read_text()))
    config['models'] = {args.model: config['models'][args.model]}
    config['models'][args.model].update(steps=300, learning_rate=1e-3)
    config.update(seeds=[151], modes=['bounded'], batch_size=64, eval_batch_size=128,
                  eval_every=25, log_every=25, cohort='exploratory_frozen_backbone_sentence_classifier',
                  selection_protocol='official_validation', training_sampling_protocol='training_roots')
    config['parent_checkpoint'] = {'path': str(path.parent / 'selected.pth'),
                                   'sha256': result['checkpoint_sha256'], 'result_path': str(path),
                                   'selected_step': result['selected_step']}
    out = args.output.resolve()
    (out / 'logs').mkdir(parents=True, exist_ok=True)
    config_path = out / (args.model + '_config.json')
    with config_path.open('x') as handle:
        json.dump(config, handle, indent=2)
        handle.write('\n')
    name = f'{args.model}_bounded_seed151'
    print(f'Adapting classifier from {path}: {result["evaluation"]}', flush=True)
    with (out / 'logs' / (name + '.log')).open('w') as log:
        subprocess.run([sys.executable, str(Path(__file__).with_name('recovery_head.py')), '--model', args.model,
                        '--config', str(config_path), '--assets', str(args.assets.resolve()),
                        '--output', str(out / name), '--seed', '151', '--device', args.device],
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    result = json.loads((out / name / 'result.json').read_text())
    print(f"Completed {name}: {result['evaluation']}; qualifies={result['meets_historical_threshold']}", flush=True)


if __name__ == '__main__':
    main()
