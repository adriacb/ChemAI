import torch
import torch.nn as nn
from typing import List, Tuple
from torch import Tensor
from torch_geometric.data import Data


class ChemDiff(nn.Module):
    def __init__(self, data:Data):
        self.data = data
        
    def forward(self, data: Data, t: int) -> Data:
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
    