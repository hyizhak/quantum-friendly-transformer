"""Freeze the three-seed validation recipe choice before final testing."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/reviewer/long_recall_train8192_summary"


def main():
    recipes, hashes = [], {}
    for name in ("cap24", "cold_long", "warm_route1", "warm_route01", "warm_answer"):
        root = ROOT / f"results/reviewer/long_recall_train8192_{name}"
        plan = json.loads((root / "plan_native_sqrt_d.json").read_text())
        assert len(plan) == 3 and all(r["status"] == "completed" for r in plan)
        config = json.loads((root / "config.json").read_text())
        assert config["evaluation_role"] == "validation" and config["test_seed"] == 4930821
        scores = []
        for seed in (11, 22, 33):
            path = root / f"native_sqrt_d_seed{seed}/result.json"
            result = json.loads(path.read_text())
            scores.append(100 * next(r["accuracy"] for r in result["evaluation"] if r["N"] == 8192))
            hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        recipes.append({"recipe": name, "root": str(root.relative_to(ROOT)),
                        "validation_mean_pct": statistics.mean(scores),
                        "validation_sd_pct": statistics.stdev(scores),
                        "seed11_pct": scores[0], "seed22_pct": scores[1], "seed33_pct": scores[2],
                        "total_updates": config["steps"] + (3000 if config.get("warm_start_root") else 0),
                        "routing_weight": config["routing_auxiliary_weight"],
                        "warm_start": bool(config.get("warm_start_root"))})
    chosen = sorted(recipes, key=lambda r:(-r["validation_mean_pct"], r["total_updates"], r["recipe"]))[0]
    uniform_root = ("results/reviewer/long_recall_train8192_warm_route1" if chosen["warm_start"]
                    else chosen["root"])
    plan = json.loads((ROOT / uniform_root / "plan_uniform.json").read_text())
    assert len(plan) == 3 and all(r["status"] == "completed" for r in plan)
    payload = {"utc": datetime.now(timezone.utc).isoformat(), "selection_split": "validation",
               "selection_rule": "Highest mean N8192 validation accuracy across all three seeds; ties use fewer total updates.",
               "all_recipes_complete": True, "final_test_used_for_selection": False,
               "reserved_test_seed": 7041937, "selected_recipe": chosen["recipe"],
               "selected_cap24_root": chosen["root"],
               "reference_cap64_root": "results/reviewer/long_recall_train8192_cap64",
               "uniform_cap24_root": uniform_root,
               "uniform_control_note": "For uniform attention the auxiliary routing loss is constant with zero parameter gradient; the three warm recipes have identical uniform-control training gradients and budgets.",
               "recipes": recipes, "validation_result_sha256": hashes}
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "selection.json").open("x") as stream:
        stream.write(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
