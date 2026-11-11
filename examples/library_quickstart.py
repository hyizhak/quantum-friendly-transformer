"""Run with an installed library; no GPU, model files, or downloads are needed."""

import io

import torch
from torch import nn

from quantum_friendly_transformer import (
    UnitNormLayerNorm,
    __version__,
    learnable_frobenius_cap,
)


def make_model():
    return nn.Sequential(
        learnable_frobenius_cap(nn.Linear(16, 16), max_gamma=2.0),
        UnitNormLayerNorm(16),
    )


def main():
    torch.manual_seed(42)
    model = make_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    inputs = torch.randn(2, 8, 16)
    targets = torch.randn(2, 8, 16)
    loss = nn.functional.mse_loss(model(inputs), targets)
    loss.backward()
    optimizer.step()
    assert model[0].weight.norm().item() <= 2.0

    checkpoint = io.BytesIO()
    torch.save(model.state_dict(), checkpoint)
    checkpoint.seek(0)
    restored = make_model()
    restored.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    torch.testing.assert_close(model(inputs), restored(inputs))
    print(f"quantum-friendly-transformer {__version__}: forward, backward, and checkpoint reload passed")


if __name__ == "__main__":
    main()
