import torch
from torch import nn, Tensor
from torch.nn.utils import parametrize

class _FrobeniusNorm(nn.Module):
    r"""
    A parametrization module that normalizes a parameter by its Frobenius norm.

    Given a weight \(\mathbf{W}\), this parametrization returns
    \[
        \frac{\mathbf{W}}{\|\mathbf{W}\|_F + \text{eps}}
    \]
    where \(\|\mathbf{W}\|_F = \sqrt{\sum_{i,j} W_{i,j}^2}\).
    """
    def __init__(self, weight: Tensor, eps: float = 1e-12):
        super().__init__()
        # We usually store eps as a buffer/attribute for numeric stability
        self.eps = eps

    def forward(self, X: Tensor) -> Tensor:
        # Compute the Frobenius norm (over all elements)
        fro_norm = torch.linalg.norm(X, ord='fro')
        # Normalize the parameter
        return X / (fro_norm + self.eps)

def frobenius_norm(
    module: nn.Module,
    name: str = "weight",
    eps: float = 1e-12
) -> nn.Module:
    r"""
    Applies Frobenius normalization to a parameter in the given module.

    The weight will be replaced by:
    \[
       \mathbf{W}_\text{FN} = \frac{\mathbf{W}}{\|\mathbf{W}\|_F + \text{eps}}
    \]

    Args:
        module (nn.Module): The module that contains the parameter to be normalized.
        name (str, optional): The name of the parameter to normalize. Default: "weight".
        eps (float, optional): A small constant for numerical stability. Default: 1e-12.

    Returns:
        The original module with a new parametrization registered to the specified
        parameter.
    """
    weight = getattr(module, name, None)
    if not isinstance(weight, Tensor):
        raise ValueError(
            f"Module '{module}' has no parameter or buffer with name '{name}'"
        )

    parametrize.register_parametrization(
        module, name, _FrobeniusNorm(weight, eps=eps)
    )
    return module
