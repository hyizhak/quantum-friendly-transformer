"""Measure columns of the paper's W matrices across all self-attention layers."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

if __package__:
    from .measure_weights import column_statistics
else:
    from measure_weights import column_statistics


def measure_column_sources(sources, output):
    rows, details = [], {}
    for name, source in sources.items():
        print(f'Measuring all-layer columns: {name}', flush=True)
        stats, layers = column_statistics(source['snapshot_path'])
        rows.append({'model': name, **stats})
        details[name] = {'source': source, 'layers': layers}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        'matrix_convention': 'S @ W; Linear weights transposed; GPT Conv1D weights unchanged',
        'method': 'float64 column L2 norms; pooled across Q/K/V and all layers; sample variance (ddof=1)',
        'gqa_weighting': 'one observation per actual output column; fewer K/V columns for grouped-query attention',
        'models': details,
        'csv_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
    }
    output.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources', type=Path, required=True, help='JSON mapping model labels to pinned local snapshots')
    parser.add_argument('--output', type=Path, default=Path(__file__).parent / 'stats/qkv_stats_corrected.csv')
    args = parser.parse_args()
    measure_column_sources(json.loads(args.sources.read_text()), args.output)


if __name__ == '__main__':
    main()
