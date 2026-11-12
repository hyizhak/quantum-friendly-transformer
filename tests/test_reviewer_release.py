from tools.build_reviewer_release import EXTRA_WEIGHTS, inventory


def test_release_inventory_retains_experiments_and_excludes_correspondence(tmp_path):
    experiment_files = {
        "src/quantum_friendly_transformer/evidence/records.zip",
        "experiments/configs/reviewer/protocol.txt",
        "results/reviewer/mqar/result.json",
        "results/reviewer/mqar/predictions.csv",
        "results/reviewer/mqar/results_table.tex",
    }
    private_files = {
        "response.tex",
        "results/reviewer/release/comment4_response.tex",
        "results/reviewer/release/response_draft_provenance.json",
        "results/reviewer/long_recall_summary/comment4_addendum.tex",
        "results/reviewer/comment4_update/comment4_only.tex",
        "results/reviewer/comment4_update/comment4_supplement.tex",
        "results/reviewer/private_correspondence/backup/tools/generator.py",
    }
    for name in experiment_files | private_files:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("private correspondence" if name in private_files else "experiment fixture")
    for name in EXTRA_WEIGHTS:
        path = tmp_path / "results/reviewer" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"checkpoint fixture")

    compact, weights, omitted = inventory(tmp_path)

    assert {path.relative_to(tmp_path).as_posix() for path in compact} == experiment_files
    assert len(weights) == len(EXTRA_WEIGHTS)
    assert omitted == []
