"""Run the declared SAM attempts, stopping after the missing model recovers."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from experiments.lib.recovery_full_dense import file_sha256
from experiments.lib.reviewer import save_json


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    plan = []
    for seed in range(901, 907):
        config = ROOT / f'experiments/configs/reviewer/recovery_sam_seed{seed}.json'
        plan.append({'seed': seed, 'config': str(config), 'config_sha256': file_sha256(config), 'status': 'planned'})
    save_json(args.output / 'attempt_plan.json', plan)
    for item in plan:
        seed = item['seed']
        run = args.output / f'2layers33m_bounded_seed{seed}'
        item['status'] = 'running'
        save_json(args.output / 'attempt_plan.json', plan)
        with (args.output / f'run_seed{seed}.log').open('w') as handle:
            outcome = subprocess.run([sys.executable, '-u', str(ROOT / 'experiments/runners/reviewer_recovery_sam.py'),
                '--config', item['config'], '--assets', str(args.assets), '--output', str(run),
                '--seed', str(seed), '--device', args.device], stdout=handle, stderr=subprocess.STDOUT)
        if outcome.returncode:
            item['status'] = 'failed'
            save_json(args.output / 'attempt_plan.json', plan)
            raise RuntimeError(f'SAM seed {seed} failed; its log is retained')
        metadata = json.loads((run / 'manifest.json').read_text())
        metadata['source_sha256'][str(Path(__file__).resolve())] = file_sha256(__file__)
        save_json(run / 'manifest.json', metadata)
        result = json.loads((run / 'result.json').read_text())
        item.update(status='complete', correct=result['evaluation']['correct'], recovered=result['meets_historical_threshold'])
        print(f'SAM seed={seed}: {item["correct"]}/872, recovered={item["recovered"]}', flush=True)
        if item['recovered']:
            for pending in plan:
                if pending['status'] == 'planned':
                    pending['status'] = 'skipped_after_recovery'
            save_json(args.output / 'attempt_plan.json', plan)
            break
        save_json(args.output / 'attempt_plan.json', plan)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    main(parser.parse_args())
