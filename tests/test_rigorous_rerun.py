"""Check split isolation, paired trials, and exact checkpoint continuation."""

import copy
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import torch

from experiments.rigorous_rerun.prepare import clean_splits, duplicate_groups
from experiments.rigorous_rerun.run import (
    capture_rng,
    epoch_order,
    jobs,
    optimizer_for,
    restore_rng,
    save_checkpoint,
    score,
    seed_everything,
    settings,
    write_json,
)
from quantum_friendly_transformer.norm_transformer.frobenius_normalized_transformer import (
    FrobeniuslyNormalizedTransformerForSequenceClassification as FrobeniusModel,
)
from quantum_friendly_transformer.norm_transformer.spectral_normalized_transformer import (
    SpectrallyNormalizedTransformerForSequenceClassification as SpectralModel,
)


def test_transitive_duplicate_groups_do_not_cross_splits_and_test_is_intact():
    # Train matches dev in raw form; dev matches test after truncation.
    groups = duplicate_groups(["a", "a", "b", "c", "d", "e"], ["x", "y", "y", "z", "u", "v"])
    selected, removed = clean_splits(["train", "validation", "test", "train", "validation", "test"], groups, [0]*6)
    assert groups[0] == groups[1] == groups[2]
    assert selected == {"train": [3], "validation": [4], "test": [2, 5]}
    assert removed == [0, 1]


def test_genomic_validation_is_from_training_groups_only_and_repeatable():
    labels = [i % 2 for i in range(100)] * 2 + [0, 1]
    groups = list(range(100)) * 2 + [100, 101]
    original = ["train"] * 200 + ["test"] * 2
    a, _ = clean_splits(original, groups, labels, make_validation=True)
    b, _ = clean_splits(original, groups, labels, make_validation=True)
    assert a == b
    assert len(a["validation"]) == 20
    assert a["test"] == [200, 201]
    assert not {groups[i] for i in a["train"]} & {groups[i] for i in a["validation"]}


def test_batch_sampling_is_independent_of_global_dropout_randomness():
    first = epoch_order(103, 42, 1)
    torch.rand(333)
    assert torch.equal(first, epoch_order(103, 42, 1))
    assert not torch.equal(first, epoch_order(103, 42, 2))
    assert first.unique().numel() == 103


def test_protocol_includes_every_variant_and_paired_parent_before_finetuning():
    root = Path(__file__).resolve().parents[1]
    protocol = json.loads((root / "experiments/configs/training/rigorous_rerun.json").read_text())
    planned = jobs(protocol)
    assert len(planned) == 105
    assert len({job["id"] for job in planned}) == 105
    seen = set()
    for job in planned:
        if job["family"] == "fine_tune":
            assert f"{job['dataset']}/{job['seed']}/single/vanilla" in seen
        seen.add(job["id"])
    assert settings(protocol, "genomic", "single")["metric"] == "f1"
    assert settings(protocol, "conll", "single")["metric"] == "accuracy"


def test_token_scores_ignore_unlabeled_subwords_and_padding():
    result = score([[1, -100, 0], [0, -100, -100]], [[1, 99, 1], [0, 98, 97]], True)
    assert result["accuracy"] == pytest.approx(2 / 3)
    assert result["labeled_elements"] == 3


def test_concurrent_metadata_writes_are_atomic_and_leave_no_partial_files(tmp_path):
    target = tmp_path / "selection_lock.json"
    payloads = [{"checkpoints": {str(i): "a" * 64 for i in range(105)}} for _ in range(16)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda value: write_json(target, value), payloads))
    assert json.loads(target.read_text()) == payloads[0]
    assert list(tmp_path.iterdir()) == [target]


@pytest.mark.parametrize("variant", ["vanilla", "sn", "fn"])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_dropout_and_parametrized_optimizer_resume_exactly(tmp_path, variant, device):
    seed_everything(42)
    args = {"d_model": 8, "nhead": 2, "d_ff": 16, "num_emb": 20, "max_seq_len": 5, "num_classes": 2}

    def make():
        if variant == "fn":
            return FrobeniusModel(**args, apply_attention_fn=True, apply_ffn_fn=True)
        return SpectralModel(**args, apply_attention_sn=variant == "sn", apply_ffn_sn=variant == "sn")

    model = make().to(device)
    cfg = {"lr": 0.001, "weight_decay": 0.001}
    opt = optimizer_for(model, cfg, "single")
    x, y = torch.tensor([[1, 2, 3, 0, 0], [3, 1, 2, 4, 0]]), torch.tensor([0, 1])
    x, y = x.to(device), y.to(device)

    def step(m, o):
        m.train()
        o.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(m(x, key_padding_mask=x == 0), y)
        loss.backward()
        o.step()
        return loss.detach()

    step(model, opt)
    path = tmp_path / "resume.pt"
    save_checkpoint(path, {"model": model.state_dict(), "optimizer": opt.state_dict(), "rng": capture_rng()})
    expected_loss = step(model, opt)
    expected = copy.deepcopy(model.state_dict())
    resumed = make().to(device)
    resumed_opt = optimizer_for(resumed, cfg, "single")
    state = torch.load(path, map_location="cpu", weights_only=False)
    resumed.load_state_dict(state["model"])
    resumed_opt.load_state_dict(state["optimizer"])
    restore_rng(state["rng"])
    actual_loss = step(resumed, resumed_opt)
    torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
    for key, value in resumed.state_dict().items():
        torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
