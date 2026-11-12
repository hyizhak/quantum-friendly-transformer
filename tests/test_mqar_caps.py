import torch

from experiments.lib.mqar import state_hash,weight_records
from experiments.lib.mqar_variance_init import MQARVarianceInit


def test_smaller_cap_preserves_raw_initial_parameters_and_changes_effective_weights():
    base={"vocab_size":128,"dimension":64,"ffn_dimension":128,"layers":2}
    torch.manual_seed(11)
    cap64=MQARVarianceInit({**base,"cap":64.})
    torch.manual_seed(11)
    cap48=MQARVarianceInit({**base,"cap":48.})
    assert state_hash(cap48.state_dict())==state_hash(cap64.state_dict())
    records=weight_records(cap48)
    targets=[r for r in records if r["capped"]]
    assert len(targets)==12
    assert all(r["effective_frobenius"]<=48 for r in targets)
    assert any(r["cap_active"] for r in targets)
    assert torch.equal(cap48.readout.weight,cap64.readout.weight)
    assert not torch.equal(cap48.blocks[0].m1.weight,cap64.blocks[0].m1.weight)


def test_smaller_cap_remains_enforced_after_an_optimizer_update():
    config={"vocab_size":128,"dimension":64,"ffn_dimension":128,"layers":2,"cap":48.}
    torch.manual_seed(22)
    model=MQARVarianceInit(config)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.003)
    inputs=torch.randint(128,(2,16))
    targets=torch.randint(128,(2,16))
    loss=torch.nn.functional.cross_entropy(model(inputs).flatten(0,1),targets.flatten())
    loss.backward()
    optimizer.step()
    assert all(r["effective_frobenius"]<=48 for r in weight_records(model) if r["capped"])
