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
    
class _FrobeniusNormWithGamma(nn.Module):
    r"""
    A parametrization module that normalizes a parameter by its Frobenius norm
    and applies a learnable scaling factor gamma that is constrained to be less than max_gamma.
    
    Given a weight \(\mathbf{W}\), this parametrization returns
    \[
        \hat{W} = \gamma \cdot \frac{\mathbf{W}}{\|\mathbf{W}\|_F + \text{eps}},
    \]
    where \(\gamma = \text{max\_gamma} \cdot \sigma(g)\) and \( g \) is an unconstrained parameter.
    """
    def __init__(self, eps: float = 1e-12, max_gamma: float = 2.0):
        super().__init__()
        self.eps = eps
        self.max_gamma = max_gamma
        # To initialize so that gamma is close to init_gamma, solve: init_gamma = max_gamma * sigmoid(g)
        # This gives: g = log(init_gamma / (max_gamma - init_gamma)).
        init_g = torch.log(torch.tensor(1 / eps))
        self.g = nn.Parameter(init_g)

    def forward(self, X: Tensor) -> Tensor:
        gamma = self.max_gamma * torch.sigmoid(self.g)
        fro_norm = torch.linalg.norm(X, ord='fro')
        return gamma * X / (fro_norm + self.eps)

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

def frobenius_norm_with_scaling(
    module: nn.Module,
    name: str = "weight",
    eps: float = 1e-12,
    max_gamma: float = 2.0
) -> nn.Module:
    r"""
    Applies Frobenius normalization with a learnable gamma scaling factor to a parameter in the given module.

    The weight will be replaced by:
    \[
        \hat{W} = \gamma \cdot \frac{\mathbf{W}}{\|\mathbf{W}\|_F + \text{eps}},
    \]
    where \(\gamma = \text{max\_gamma} \times \sigma(g)\), with \( g \) a learnable parameter
    that is initialized so that \(\gamma\) starts close to `init_gamma`.

    Args:
        module (nn.Module): The module that contains the parameter to be normalized.
        name (str, optional): The name of the parameter to normalize. Default: "weight".
        eps (float, optional): A small constant for numerical stability. Default: 1e-12.
        init_gamma (float, optional): The initial scaling factor for gamma (must be less than max_gamma). Default: 1.0.
        max_gamma (float, optional): The maximum allowed scaling factor for gamma. Default: 1.0.

    Returns:
        The original module with the new parametrization registered to the specified parameter.
    """
    weight = getattr(module, name, None)
    if not isinstance(weight, Tensor):
        raise ValueError(f"Module '{module}' has no parameter or buffer with name '{name}'")
    
    # Register the parametrization with learnable gamma scaling
    parametrize.register_parametrization(
        module, name, _FrobeniusNormWithGamma(eps=eps, max_gamma=max_gamma)
    )
    return module