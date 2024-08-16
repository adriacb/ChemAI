import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from tqdm import tqdm
from typing import Tuple

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), 
                                 os.pardir, 
                                 os.pardir
                                 ))) # Add llm to path

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
    t_quant = (t - zero_point) / scale
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
    for name, param in tqdm(model.named_parameters()):
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
    for name, param in tqdm(model.named_parameters()):
        param.data = dequantize(param.data, states[name])
    return model

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quantization.")
    parser.add_argument(
        "--model",
        type=str,
        default="model.pth",
        help="Path to the model.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="model_quantized.pth",
        help="Path to the output model.",
    )
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    # check if the model exists
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model not found: {args.model}")
    # load the model
    model = torch.load(args.model)
    print(f"Model loaded from {args.model}")
    num_bytes = get_memory_footprint(model)
    print(f"Memory footprint: {num_bytes} bytes")
    model, states = quantize_model(model)
    torch.save(model, args.output)
    torch.save(states, args.output + ".states")
    print(f"Model quantized and saved to {args.output}")
    num_bytes = get_memory_footprint(model)
    print(f"Memory footprint: {num_bytes} bytes")

if __name__ == "__main__":
    main()