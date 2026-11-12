from tools.build_research_bundle import inventory


def test_release_inventory_retains_experiments_and_excludes_correspondence(tmp_path):
    experiment_files = {
        "src/quantum_friendly_transformer/evidence/records.zip",
        "experiments/configs/protocol.txt",
        "results/mqar/result.json",
        "results/mqar/predictions.csv",
        "results/mqar/results_table.tex",
    }
    private_files = {
        "response.tex",
        "results/release/response.tex",
        "results/release/response_draft_provenance.json",
        "results/long_recall_summary/addendum.tex",
        "results/correspondence/notes.tex",
        "results/correspondence/supplement.tex",
        "results/private_correspondence/backup/tools/generator.py",
    }
    for name in experiment_files | private_files:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("private correspondence" if name in private_files else "experiment fixture")
    checkpoints = ["mqar/run/final.pth", "recovery/run/selected.pth"]
    for name in checkpoints:
        path = tmp_path / "results" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"checkpoint fixture")

    compact, weights, omitted = inventory(tmp_path, tmp_path / "results", include_checkpoints=True)

    assert {path.relative_to(tmp_path).as_posix() for path in compact} == experiment_files
    assert len(weights) == len(checkpoints)
    assert omitted == []
