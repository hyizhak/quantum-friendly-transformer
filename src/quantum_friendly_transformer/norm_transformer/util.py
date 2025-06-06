from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.parametrizations import spectral_norm
from .frobenius_norm import frobenius_norm

def apply_spectral_norm(module, layers=(nn.Linear,)):
    """
    Recursively apply spectral normalization to the specified layer types.
    
    Args:
        module (nn.Module): The model or submodule to modify.
        layers (tuple): Layer classes to apply SN to.
    """

    def SN(module):
        if isinstance(module, layers):
            module = spectral_norm(module)

    module.apply(SN)

def apply_frobenius_norm(module, layers=(nn.Linear,)):
    """
    Recursively apply frobenius normalization to the specified layer types.
    
    Args:
        module (nn.Module): The model or submodule to modify.
        layers (tuple): Layer classes to apply FN to.
    """

    def FN(module):
        if isinstance(module, layers):
            module = frobenius_norm(module)

    module.apply(FN)

def _check_arg_device(x: Optional[torch.Tensor]) -> bool:
    if x is not None:
        return x.device.type in [
            "cpu",
            "cuda",
            torch.utils.backend_registration._privateuse1_backend_name,
        ]
    return True


def _arg_requires_grad(x: Optional[torch.Tensor]) -> bool:
    if x is not None:
        return x.requires_grad
    return False


def _is_make_fx_tracing():
    if not torch.jit.is_scripting():
        torch_dispatch_mode_stack = (
            torch.utils._python_dispatch._get_current_dispatch_mode_stack()
        )
        return any(
            type(x) == torch.fx.experimental.proxy_tensor.ProxyTorchDispatchMode
            for x in torch_dispatch_mode_stack
        )
    else:
        return False

