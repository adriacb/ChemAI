import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from tqdm import tqdm

def get_float32_dtype(self):
    return torch.float32

# model.dtype = property(get_float32_dtype)

def syze_in_bytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()

def get_memory_footprint(model: nn.Module,
                          return_buffers: bool = True) -> int:
    """Return the memory footprint of the model in bytes."""

    mem = sum([p.nelement() * p.element_size() for p in model.parameters()])
    if return_buffers:
        mem += sum([p.nelement() * p.element_size() for p in model.buffers()])
    return mem

def quantize(t: torch.Tensor) -> Tuple[torch.Tensor, Tuple[float, float]]:
    """Quantize a tensor using min-max quantization.

    Args:
    - model: nn.Module
    - t: torch.Tensor

    Returns:
    - t_quant: torch.Tensor
    - state: Tuple[float, float]"""
    # find the min and max values
    min_val, max_val = t.min(), t.max()

    # determine the scale and zero point
    scale = (max_val - min_val) / 255
    zero_point = min_val

    # quantize the tensor
    t_quant = torch.round((t - zero_point) / scale)
    t_quant = torch.clamp(t_quant, 0, 255)

    # keep track of the scale and zero point
    state = (scale, zero_point)

    # cast the tensor to uint8
    t_quant = t_quant.type(torch.uint8)
    return t_quant, state

def dequantize(t_quant: torch.Tensor, state: Tuple[float, float]) -> torch.Tensor:
    """Dequantize a tensor using min-max dequantization.

    Args:
    - t_quant: torch.Tensor
    - state: Tuple[float, float]

    Returns:
    - t: torch.Tensor"""
    scale, zero_point = state
    return t_quant.to(torch.float32) * scale + zero_point

def quantize_model(model: nn.Module) -> Tuple[nn.Module, dict]:
    """Quantize the model using min-max quantization.
    
    Args:
    - model: nn.Module

    Returns:
    - model: nn.Module
    - states: dict"""
    states = {}
    for name, param in model.named_parameters():
        param.requires_grad = False # disable gradient computation
        param.data, state = quantize(param.data)
        states[name] = state
    
    return model, states

def dequantize_model(model: nn.Module, states: dict) -> nn.Module:
    """Dequantize the model using min-max dequantization.
    
    Args:
    - model: nn.Module
    - states: dict

    Returns:
    - model: nn.Module"""
    for name, param in model.named_parameters():
        param.data = dequantize(param.data, states[name])
    return model