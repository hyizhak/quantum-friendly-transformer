"""Replay all completed exploratory search attempts, including large backbones."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

from experiments.runners import validate_recovery_extension as audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--large', action='store_true')
    parser.add_argument('--low-rank', action='store_true')
    parser.add_argument('--full-dense', action='store_true')
    parser.add_argument('--replay', action='store_true')
    parser.add_argument('--only-seed', action='append', type=int,
                        help='Audit/replay only this recorded seed or candidate ID (repeatable).')
    args = parser.parse_args()
    if args.large or args.low_rank or args.full_dense:
        from experiments.runners import reviewer_recovery_extension as core
        from experiments.runners.reviewer_recovery_large import build_model, LargeAttentionInputMonitor
        if args.low_rank:
            from experiments.runners.reviewer_recovery_low_rank import build_model
        if args.full_dense:
            from experiments.runners import reviewer_recovery_full_dense as full_dense

            def build_model(assets, name, config, seed, mode, device):
                assert name == 'mistral7b' and mode == 'bounded'
                model, caps, adapters, _, _ = full_dense.build_model(
                    assets, config, seed, [device], initialize_parent=False)
                return model, caps, adapters
        core.build_model = build_model
        core.AttentionInputMonitor = LargeAttentionInputMonitor
        original_measure = audit.measure_attention

        def measure(model, data, device, limit=128, batch_size=4):
            # The queried 128-token context lies within Mistral's sliding window.
            config = model.config
            config.attention_layers = ['global'] * config.num_hidden_layers
            return original_measure(model, data, device, limit=limit, batch_size=batch_size)

        audit.measure_attention = measure
    audit.validate(argparse.Namespace(config=args.config, results=args.results, assets=args.assets,
                                      device=args.device, replay=args.replay, allow_partial=True,
                                      only_seeds=args.only_seed,
                                      output_suffix='_' + args.config.stem))


if __name__ == '__main__':
    main()
