import torch
import numpy as np
from typing import Tuple
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Data

from .scheduler import beta_schedule

def to_torch_const(x):
    x = torch.from_numpy(x).float()
    x = nn.Parameter(x, requires_grad=False)
    return x

class Diffusion:
    def __init__(self, timesteps:int):
        self.timesteps = timesteps
        self.scaling = 1.0                      # Scaling factor for the noise.

    @staticmethod
    def get_betas(timesteps:int, **kwargs) -> torch.Tensor:
        """
        Returns the beta schedule tensor based on the given type.
        
        Args:
            timesteps (int): The number of time steps.
            type (str): The type of beta schedule.
        
        Returns:
            torch.Tensor: The beta schedule tensor.
        """
        print
        if kwargs["type"] == 'advance':
            betas = beta_schedule(timesteps, **kwargs)
        elif kwargs["type"] == 'segment':
            betas = beta_schedule(timesteps, **kwargs)
        else:
            raise ValueError(f"Invalid beta schedule type: {kwargs['type']}")
        return betas
    
    def qrt(self, 
            r: torch.Tensor, 
            t: int, 
            beta_config:dict={
                            'type': 'advance',
                            'scale_start': 0.9999,
                            'scale_end': 0.0001,
                            'width': 3}
        ) -> torch.Tensor:
        """
        Adds noise to the ATOMIC POSITIONS, the noise is modeled as a Gaussian distribution.
        
        q(r^{t}_i | r^{t-1}_i) := \mathcal{N}(r^{t}_i | \sqrt{1-\beta^{t} r^{t-1}_i}, \beta^{t}\mathbb{I})
        - $\mathbb{I}^{3x3}$ is the identity matrix.
        - $\beta^t \in [0,1]$ (noise scaling schedule).

        Args:
            r (torch.Tensor): The atomic positions tensor.
            t (int): The current time step.
            beta_config (dict): The beta configuration.
        
        Returns:
            torch.Tensor: The noisy atomic positions tensor.
        """
        betas = self.get_betas(self.timesteps, **beta_config)
        alphas = 1. - betas
        alphas_hat = torch.cumprod(alphas, dim=0)
        alpha_t = alphas_hat[t].reshape(1, -1)
        noise = torch.randn_like(r)
        noisy_r = torch.sqrt(alpha_t) * r + torch.sqrt(1 - alpha_t) * noise
        return noisy_r

    def qat(self, a: torch.Tensor, t: int, 
            beta_config:dict={
                            'type': 'advance',
                            'scale_start': 0.9999,
                            'scale_end': 0.0001,
                            'width': 3}
            ) -> torch.Tensor:
        """
        Adds noise to the ATOM TYPES, the noise is modeled as a Categorical distribution.

        q(a^{t}_i | a^{t-1}_i) := \mathcal{C}(a^{t}_i | (1-\beta^{t}) a^{t-1}_i, \beta^{t}\mathbb{1}_k)
        - $\mathbb{1}_k$ represents a one-hot vector with one at the kth position and all the others zeros.
        - $\beta^t \in [0,1]$ (noise scaling schedule).

        Args:
            a (torch.Tensor): The atom types tensor.
            t (int): The current time step.
            beta_config (dict): The beta configuration.
        
        Returns:
            torch.Tensor: The noisy atom types tensor.
        """
        betas = self.get_betas(self.timesteps, **beta_config)
        alphas = 1. - betas
        alphas_hat = torch.cumprod(alphas, dim=0)
        alpha_t = alphas_hat[t].reshape(1, -1)
        noise = torch.randint_like(a, 0, a.shape[1])
        noisy_a = (1 - alpha_t) * a + alpha_t * noise
        return noisy_a
    


    def forward(self, data: dict, t: int) -> dict:
        """
        Adds noise to the molecule data based on the given noise schedule at time step t.
        
        For atom types and bond types we represent them using categorical distributions.

        ```
        - $q(r^{t}_i | r^{t-1}_i) := \mathcal{N}(r^{t}_i | \sqrt{1-\beta^{t} r^{t-1}_i}, \beta^{t}\mathbb{I})$
        - $q(a^{t}_i | a^{t-1}_i) := \mathcal{C}(a^{t}_i | (1-\beta^{t}) a^{t-1}_i, \beta^{t}\mathbb{1}_k)$ 
        - $q(b^{t}_{ij} | b^{t-1}_{ij}) := \mathcal{C}(b^{t}_{ij} | (1-\beta^{t}) b^{t-1}_{ij}, \beta^{t}\mathbb{1}_k)$        
        ```
        """
        noisy_data = {}
        
        # Apply noise to positions
        pos = data['pos']
        noisy_pos = self.qrt(pos, t)
        noisy_data['pos'] = noisy_pos

        # Apply noise to atom types (Categorical noise)
        atom_types = data['x']
        noisy_atom_types = self.qat(atom_types, t)
        noisy_data['x'] = noisy_atom_types

        # # Apply noise to bond types (Categorical noise)
        # bond_types = data['edge_attr']
        # noisy_bond_types = self.qat(bond_types, t)
        # noisy_data['edge_attr'] = noisy_bond_types

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