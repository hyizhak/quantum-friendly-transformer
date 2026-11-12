"""Plot recorded validation trajectories from the completed guided protocols."""

import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    args = parser.parse_args()
    root = args.results
    summary = json.loads((root / 'recovery_guided_summary.json').read_text())
    by_name = {row['model']: row for row in summary['rows']}

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(10.5, 7), constrained_layout=True)
    panels = [
        (axes[0, 0], '2layers33m', 'Two-layer continuation: matched teacher control', [
            ('recovery_trajectory/2layers33m_bounded_seed611', 'Gold only', '#3569a7'),
            ('recovery_guided/2layers33m_bounded_seed611', 'With distillation', '#bd6532')]),
        (axes[0, 1], '2layers33m', 'Two-layer restarts from pretrained initialization', [
            (f'recovery_guided_restart/2layers33m_bounded_seed{seed}', f'Seed {seed}', color)
            for seed, color in zip([811, 812, 813], ['#3569a7', '#bd6532', '#6b8b41'])]),
        (axes[1, 0], 'olmo1b', 'OLMo-1B: full-parameter guided continuation', [
            ('recovery_guided/olmo1b_bounded_seed611', 'With distillation', '#3569a7')]),
        (axes[1, 1], 'mistral7b', 'Mistral-7B: full-parameter guided continuation', [
            ('recovery_guided/mistral7b_bounded_seed611', 'With distillation', '#3569a7')]),
    ]
    for axis, name, title, runs in panels:
        for relative, label, color in runs:
            path = root / relative
            assert (path / 'result.json').exists(), f'Incomplete run: {path}'
            rows = list(csv.DictReader((path / 'selection.csv').open()))
            for variant, style in [('live', '-'), ('average', ':')]:
                selected = [row for row in rows if row['variant'] == variant]
                if not selected:
                    continue
                axis.plot([int(row['step']) for row in selected],
                          [100 * int(row['correct']) / int(row['examples']) for row in selected],
                          color=color, linestyle=style, linewidth=1.6,
                          label=label if variant == 'live' else '_nolegend_')
        row = by_name[name]
        axis.axhline(100 * row['previous_correct'] / row['examples'], color='#777777',
                     linewidth=1, linestyle='-.', label='Previous best')
        axis.axhline(100 * row['threshold'], color='#9b2834', linewidth=1.2,
                     linestyle='--', label='Recovery threshold')
        axis.set(title=title, xlabel='Optimizer update', ylabel='Validation accuracy (%)')
        axis.grid(alpha=.2)
        axis.legend(fontsize=7, loc='best')
    figure.suptitle('Guided bounded recovery: recorded training trajectories\n'
                    'Solid: live parameters; dotted: trajectory average. Validation selects checkpoints.', fontsize=11)
    destination = root / 'figures'
    destination.mkdir(exist_ok=True)
    for extension in ['png', 'pdf']:
        figure.savefig(destination / f'recovery_guided.{extension}', dpi=180)
    plt.close(figure)
    print(destination / 'recovery_guided.pdf')


if __name__ == '__main__':
    main()
