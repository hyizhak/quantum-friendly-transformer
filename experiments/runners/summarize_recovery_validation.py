"""Check validation coverage and current source hashes for completed bounded attempts."""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path('results/reviewer'))
    args = parser.parse_args()
    root = args.results.resolve()
    evidence = {}
    for pattern in ['recovery*/artifact_validation*.json', 'recovery*/checkpoint_replay*.json']:
        for path in root.glob(pattern):
            payload = json.loads(path.read_text())
            for report in payload.get('reports', []):
                evidence.setdefault(str(path.parent / report['run']), []).append((str(path), report))
    sources, rows, missing, mismatches = {}, [], [], []
    for path in sorted(root.glob('recovery*/*_bounded_seed*/manifest.json')):
        metadata = json.loads(path.read_text())
        result = json.loads((path.parent / 'result.json').read_text())
        if result['smoke']:
            continue
        for source, expected in metadata['source_sha256'].items():
            if source not in sources:
                sources[source] = hashlib.sha256(Path(source).read_bytes()).hexdigest()
            if sources[source] != expected:
                mismatches.append({'manifest': str(path), 'source': source})
        records = evidence.get(str(path.parent), [])
        verified = any(r.get('source_hashes_match') and r.get('checkpoint_hash_matches')
                       and r.get('matrix_caps_checked', 0) > 0 for _, r in records)
        replayed = any(r.get('replayed') and r.get('all_predictions_match') for _, r in records)
        if not verified:
            missing.append(str(path.parent))
        rows.append({'run': str(path.parent), 'artifact_verified': verified,
                     'fresh_predictions_reproduced': replayed,
                     'evidence_files': sorted({p for p, _ in records})})
    report = {'completed_bounded_attempts': len(rows), 'artifact_verified_attempts': sum(r['artifact_verified'] for r in rows),
              'fresh_prediction_replays': sum(r['fresh_predictions_reproduced'] for r in rows),
              'distinct_source_files_checked': len(sources), 'uncovered_completed_attempts': missing,
              'current_source_hash_mismatches': mismatches, 'rows': rows,
              'scope': 'Completed new bounded attempts only; original archived results and incomplete jobs are excluded. Artifact audit and fresh inference replay are distinguished.'}
    (root / 'recovery_validation_coverage.json').write_text(json.dumps(report, indent=2) + '\n')
    (root / 'recovery_source_consistency.json').write_text(json.dumps({
        'completed_manifests': len(rows), 'distinct_source_files': len(sources),
        'source_mismatches': mismatches,
    }, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'rows'}, indent=2))
    if missing or mismatches:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
