import torch
import numpy as np
from typing import Tuple
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Data

def segment_schedule(timesteps: int, time_segment: list, segment_diff: list) -> torch.Tensor:
    """
    Generates a schedule for beta values over different segments.

    Parameters:
        timesteps (int): Total number of timesteps.
        time_segment (list): List of segment lengths.
        segment_diff (list): List of dictionaries with parameters for each segment.

    Returns:
        torch.Tensor: Beta values for the entire schedule.
    """
    assert np.sum(time_segment) == timesteps

    alphas_cumprod = []

    for i in range(len(time_segment)):
        time_this = time_segment[i] + 1
        params = segment_diff[i]
        _, alphas_this = advance_schedule(time_this, **params, return_alphas_bar=True)
        alphas_cumprod.extend(alphas_this[1:])

    alphas_cumprod = np.array(alphas_cumprod)
    
    alphas = np.zeros_like(alphas_cumprod)
    alphas[0] = alphas_cumprod[0]
    alphas[1:] = alphas_cumprod[1:] / alphas_cumprod[:-1]
    betas = 1 - alphas
    betas = np.clip(betas, 0, 1)

    return torch.tensor(betas, dtype=torch.float32)

def sigmoid(x: float) -> float:
    """
    Sigmoid function.
    """
    return 1 / (1 + np.exp(-x))

def advance_schedule(
        timesteps: int, 
        scale_start:float, 
        scale_end: float,
        width: int,
        return_alphas_bar: bool = False
        ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates a schedule for beta values over different segments.
    """
    k = width
    A0 = scale_end
    A1 = scale_start

    a = (A0-A1)/(sigmoid(-k) - sigmoid(k))
    b = 0.5 * (A0 + A1 - a)

    x = np.linspace(-1, 1, timesteps)
    y = a * sigmoid(- k * x) + b
    # print(y)
    
    alphas_cumprod = y 
    alphas = np.zeros_like(alphas_cumprod)
    alphas[0] = alphas_cumprod[0]
    alphas[1:] = alphas_cumprod[1:] / alphas_cumprod[:-1]
    betas = 1 - alphas
    betas = np.clip(betas, 0, 1)
    if return_alphas_bar:
        return torch.tensor(betas, dtype=torch.float32), torch.tensor(alphas, dtype=torch.float32)
    return torch.tensor(betas, dtype=torch.float32)

def forward_diffusion(data: dict, t: int, timesteps:int) -> dict:
    """
    Adds noise to the molecule data based on the given noise schedule at time step t.
    
    Args:
        data (dict): The molecule data containing atom types, bond types, and positions.
        t (int): The current time step.
        betas (torch.Tensor): The noise scaling schedule tensor.
    
    Returns:
        dict: The noisy molecule data.
    """
   
    noisy_data = {}
    
    # Apply noise to positions
    betas = beta_schedule(timesteps, scale_start=0.9999, scale_end=0.0001, type='advance', width=3)
    alphas = 1 - betas
    alpha_hat = torch.cumprod(alphas, dim=0)
    pos = data['pos']
    noise = torch.randn_like(pos)
    alpha_t = alpha_hat[t]
    mean = torch.sqrt(alpha_t) * pos
    variance = torch.sqrt(1 - alpha_t) * noise
    noisy_pos = mean + variance
    noisy_data['pos'] = noisy_pos
    
    # Apply noise to atom types (Categorical noise)
    betas = beta_schedule(timesteps, scale_start=0.9999, scale_end=0.0001, type='advance', width=3)
    alphas = 1 - betas
    alpha_hat = torch.cumprod(alphas, dim=0)
    atom_types = data['x']
    alpha_t = alpha_hat[t].reshape(1, -1)
    noise = torch.rand_like(atom_types)  # Use uniform noise for categorical data
    noisy_atom_types = torch.sqrt(alpha_t) * atom_types + torch.sqrt(1 - alpha_t) * noise
    noisy_atom_types = F.softmax(noisy_atom_types, dim=-1)
    noisy_data['x'] = noisy_atom_types
    
    # Apply noise to bond types (Categorical noise)
    betas = beta_schedule(
        timesteps, 
        time_segment=[600, 400], 
        segment_diff=[
            {'scale_start': 0.9999, 'scale_end': 0.001, 'width': 3}, 
            {'scale_start': 0.001, 'scale_end': 0.0001, 'width': 2}],
            type = 'segment'
            )
    bond_types = data['edge_attr']
    alphas = 1 - betas
    alpha_hat = torch.cumprod(alphas, dim=0)
    alpha_t = alpha_hat[t].reshape(1, -1)
    noise = torch.rand_like(bond_types)  # Use uniform noise for categorical data
    noisy_bond_types = torch.sqrt(alpha_t) * bond_types + torch.sqrt(1 - alpha_t) * noise
    noisy_bond_types = F.softmax(noisy_bond_types, dim=-1)
    noisy_data['edge_attr'] = noisy_bond_types
    
    # Preserve other data
    noisy_data['edge_index'] = data['edge_index']
    noisy_data['z'] = data['z']
    noisy_data['idx'] = data['idx']
    noisy_data['complete_edge_index'] = data['complete_edge_index']
    noisy_data['y'] = data['y']
    noisy_data['name'] = data['name']
    noisy_data['smiles'] = data['smiles']
    noisy_data['bond_matrix'] = data['bond_matrix']
    
    return Data(**noisy_data)

def beta_schedule(
        timesteps: int, 
        type: str = 'linear', 
        **kwargs: float
                  ) -> torch.Tensor:
    """
    Returns the value of beta for the given timesteps.
    Amount of noise that is being applied at each time step.

    Parameters:

        timesteps (int): The total number of time steps.
        type (str): The type of schedule ('linear' or 'cosine').

    Returns:
        torch.Tensor: The value of beta for each time step.
    """
    if type == 'linear':
        return torch.linspace(kwargs['beta_min'], kwargs['beta_max'], timesteps)
    elif type == 'cosine':
        #  as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
        steps = timesteps + 1
        x = np.linspace(0, steps, steps)
        alphas_cumprod = np.cos(((x / steps) + 0.008) / (1 + 0.008) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.tensor(betas, dtype=torch.float32)
    elif type == 'advance':
        return advance_schedule(scale_start=kwargs['scale_start'], scale_end=kwargs['scale_end'],
                                width=kwargs['width'], timesteps=timesteps, return_alphas_bar=False)
    elif type == 'segment':
        return segment_schedule(timesteps, kwargs['time_segment'], kwargs['segment_diff'])
    else:
        raise ValueError(f"Unknown schedule type: {type}. Supported types are 'linear' and 'cosine'.")
