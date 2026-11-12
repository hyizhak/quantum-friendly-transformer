"""Try teacher-guided two-layer training from bounded pretrained initialization."""

import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

import torch
from experiments.lib.recovery_full_dense import file_sha256
from experiments.lib.experiment import save_json
from experiments.runners import recovery_extension as core


def main(args):
    base = json.loads(args.template.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    initializations = args.output / 'initializations'
    initializations.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    for seed in args.seeds:
        config = copy.deepcopy(base)
        config.update(seeds=[seed], cohort='exploratory_teacher_guided_pretrained_restart',
                      average_start=3000, average_every=100, cycle_steps=500)
        config['models']['2layers33m'].update(steps=5000, learning_rate=5e-5)
        config['initialization_policy'] = 'Original pinned pretrained backbone with bounded parametrizations and a fresh seeded classifier; no preceding supervised optimization.'
        model, _, _ = core.build_model(args.assets, '2layers33m', config, seed, 'bounded', torch.device('cpu'))
        bootstrap = initializations / f'2layers33m_seed{seed}.pth'
        torch.save({'model_state_dict': model.state_dict(), 'model': '2layers33m', 'mode': 'bounded',
                    'seed': seed, 'step': 0, 'config': config}, bootstrap)
        del model
        digest = file_sha256(bootstrap)
        bootstrap_result = bootstrap.with_suffix('.json')
        save_json(bootstrap_result, {'name': '2layers33m', 'mode': 'bounded', 'checkpoint_sha256': digest,
                  'configured_denominator_floor': config['models']['2layers33m']['alpha_s']**2 * config['cap']**2,
                  'initialization_only': True, 'supervised_optimizer_steps': 0,
                  'source_sha256': {str(Path(__file__).resolve()): file_sha256(__file__)}})
        config['parent_checkpoint'] = {'path': str(bootstrap), 'sha256': digest,
                                      'result_path': str(bootstrap_result), 'selected_step': 0}
        config_path = args.output / f'config_seed{seed}.json'
        save_json(config_path, config)
        run = args.output / f'2layers33m_bounded_seed{seed}'
        command = [sys.executable, '-u', str(ROOT / str(Path(__file__).with_name('recovery_guided.py'))),
                   '--config', str(config_path), '--assets', str(args.assets), '--model', '2layers33m',
                   '--seed', str(seed), '--devices', args.device, '--output', str(run), '--teacher', str(args.teacher)]
        with (args.output / f'run_seed{seed}.log').open('w') as log:
            outcome = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        if outcome.returncode:
            raise RuntimeError(f'Seed {seed} failed; retained {log.name}')
        metadata = json.loads((run / 'manifest.json').read_text())
        metadata.update(initialization=config['initialization_policy'],
                        seed_role='initial classifier and training order')
        metadata['source_sha256'][str(Path(__file__).resolve())] = file_sha256(__file__)
        metadata['limitations'].append('This run begins from a zero-update bounded pretrained bootstrap, not from a validation-selected supervised parent.')
        save_json(run / 'manifest.json', metadata)
        result = json.loads((run / 'result.json').read_text())
        result.update(initialization=config['initialization_policy'], seed_role=metadata['seed_role'])
        save_json(run / 'result.json', result)
        print(f'seed={seed} correct={result["evaluation"]["correct"]}/872 recovered={result["meets_historical_threshold"]}', flush=True)
        if result['meets_historical_threshold']:
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--template', type=Path, required=True)
    parser.add_argument('--teacher', type=Path, required=True)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', nargs='+', type=int, default=[811, 812, 813])
    parser.add_argument('--device', default='cuda:0')
    main(parser.parse_args())
