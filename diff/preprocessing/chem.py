import os
import yaml
import torch
from pydantic import BaseModel, Field
from typing import Tuple, List, Dict
from rdkit import Chem
from rhea.chem.molecule import Molecule
from rhea.chem.structure import Structure
from torch_geometric.data import Data

class Data2(BaseModel):
    """
    Data class for PyTorch
    """
    x: torch.Tensor = Field(..., description="Node features")
    edge_index: torch.Tensor = Field(..., description="Edge connectivity")
    edge_attr: torch.Tensor = Field(..., description="Edge features")
    y: torch.Tensor = Field(..., description="Target")
    pos: torch.Tensor = Field(..., description="Atomic coordinates")
    idx: torch.Tensor = Field(..., description="Index of the molecule")
    name: str = Field(..., description="Molecule name")
    z: torch.Tensor = Field(..., description="Atomic numbers")
    complete_edge_index: torch.Tensor = Field(..., description="Complete graph connectivity")
    
    class Config:
        arbitrary_types_allowed = True
    
    def __str__(self):
        return f"Data(x={self.x}, edge_index={self.edge_index}, edge_attr={self.edge_attr}, y={self.y}, pos={self.pos}, idx={self.idx}, name={self.name}, z={self.z}, complete_edge_index={self.complete_edge_index})"

ATOM_TYPES = ['C', 'O', 'N', 'H', 'F', 'Cl', 'Br', 'I', 'P', 'S']

# Example predefined bond types
BOND_TYPES = [Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE, Chem.BondType.AROMATIC]

# current directory
current_dir = os.path.dirname(os.path.realpath(__file__))

with open(os.path.join(current_dir, os.pardir, "config", "mapping.yml"), "r") as f:
    mapping = yaml.safe_load(f)
    MAP_ATOM_SYMBOLS = mapping["MAP_ATOM_SYMBOLS"]
    MAP_HYBRIDIZATION = mapping["MAP_HYBRIDIZATION"]
    MAP_CHIRALITY = mapping["MAP_CHIRALITY"]
    MAP_BOND_TYPES = mapping["MAP_BOND_TYPES"]                                                  # https://www.rdkit.org/new_docs/cppapi/classRDKit_1_1Bond.html
    
with open(os.path.join(current_dir, os.pardir, "config", "features.yml"), "r") as f:
    features = yaml.safe_load(f)
    ATOM_FEATURES = features["ATOM_FEATURES"]
    BOND_FEATURES = features["BOND_FEATURES"]


def get_atom_type_one_hot(atom: Chem.Atom) -> torch.Tensor:
    """
    Returns a one-hot encoded vector for the atom type.
    
    Args:
        atom (Chem.Atom): RDKit atom object.
    
    Returns:
        torch.Tensor: One-hot encoded atom type vector.
    """
    atom_symbol = atom.GetSymbol()
    one_hot_vector = torch.zeros(len(ATOM_TYPES), dtype=torch.float32)
    if atom_symbol in ATOM_TYPES:
        one_hot_vector[ATOM_TYPES.index(atom_symbol)] = 1.0
    return one_hot_vector

def get_bond_type_one_hot(bond: Chem.Bond) -> torch.Tensor:
    """
    Returns a one-hot encoded vector for the bond type.
    
    Args:
        bond (Chem.Bond): RDKit bond object.
    
    Returns:
        torch.Tensor: One-hot encoded bond type vector.
    """
    bond_type = bond.GetBondType()
    one_hot_vector = torch.zeros(len(BOND_TYPES), dtype=torch.float32)
    if bond_type in BOND_TYPES:
        one_hot_vector[BOND_TYPES.index(bond_type)] = 1.0
    return one_hot_vector

def protein_to_Data(protein:Structure, features:Tuple[str|List[str]]) -> torch.Tensor:
    """
    Convert a protein into a torch geometric `Data` object.
    
    
    """
    
    raise NotImplementedError("The protein_to_point_cloud function is not implemented yet.")

def get_atom_features(atom:Chem.rdchem.Atom, features:Tuple[str|List[str]]) -> torch.Tensor:
    """
    Get atom features from a molecule.

    Args:
        mol (Molecule): The molecule.
        features (Tuple[str | List[str]], optional): The features to consider. Defaults to [""].

    Returns:
        torch.Tensor: The atom features.
    """
    # the methods normally starts with "Get", so we can use this to get the features using getattr
    
    if isinstance(features, str):
        features = [features]
    
    feat = []
    for feature in features:
        res = getattr(atom, f"Get{feature}")()
        if feature == "Hybridization":
            res = MAP_HYBRIDIZATION[res.name]
        elif feature == "Symbol":
            res = MAP_ATOM_SYMBOLS[res]
        elif feature == "ChiralTag":
            res = MAP_CHIRALITY[res.name]
        if isinstance(res, list):
            feat.extend(res)
        else:
            feat.append(res)
    
    return torch.tensor(feat, dtype=torch.float32)
        
def get_bond_features(bond:Chem.rdchem.Bond, features:Tuple[str|List[str]]) -> torch.Tensor:
    """
    Get bond features from a molecule.

    Args:
        bond (Chem.rdchem.Bond): The bond.
        features (Tuple[str | List[str]], optional): The features to consider. Defaults to [""].

    Returns:
        torch.Tensor: The bond features.
    """
    if isinstance(features, str):
        features = [features]
    
    feat = []
    for feature in features:
        res = getattr(bond, f"Get{feature}")()
        if isinstance(res, Chem.rdchem.Atom):
            res = int(res.GetIdx())
        if isinstance(res, list):
            feat.extend(res)
        else:
            feat.append(res)
    
    return torch.tensor(feat, dtype=torch.float32)

def molecule_to_Data(mol: Molecule, 
                     idx:int = 0,
                     atomfeatures:Tuple[str|List[str]] = ATOM_FEATURES, 
                     bondfeatures:Tuple[str|List[str]] = BOND_FEATURES,
                     target:torch.Tensor = torch.tensor([0], dtype=torch.float32)
                     ) -> Data:
    """
    Converts a molecule into a torch geometric `Data` object.
    
    The structure of the `Data` object is as follows:
    ```
            {
                'x': torch.Tensor,                   # node features  (e.g. formal charge, membership to aromatic rings, chirality, …)
                'edge_index': torch.Tensor,          # edges between atoms derived from covalent bonds (source_n, target_n)
                'edge_attr': torch.Tensor,           # bond features (e.g. bond type, ring-membership, …)
                'y': torch.Tensor,                   # target
                'pos': torch.Tensor,                 # atomic coordinates
                'idx': torch.Tensor,                 # index of the molecule
                'name': str,                         # molecule name
                'z': torch.Tensor,                   # atomic numbers
                'complete_edge_index': torch.Tensor  # complete graph connectivity
            }
    ```
    
    Args:
        mol (Molecule): The molecule.
        atomfeatures (Tuple[str|List[str]], optional): The atom features to consider. Defaults to ATOM_FEATURES.
        bondfeatures (Tuple[str|List[str]], optional): The bond features to consider. Defaults to BOND_FEATURES.
        target (torch.Tensor, optional): The target. Defaults to torch.tensor([0], dtype=torch.float32).
    
    Returns:
        Dict[str, torch.Tensor]: The molecule as a `Data` object.
    """

    data = dict()
    data['name'] = mol.get_name() if mol.get_name() is not None else "Molecule_{}".format(idx)
    data['smiles'] = Chem.MolToSmiles(mol.molecule)
    coordinates = mol.get_coordinates()
    data['pos'] = torch.tensor(coordinates, dtype=torch.float32)
    
    x = []                                                                                          # X belong to R^m*d is a matrix of m atoms with d features each
    edge_index = []                                                                                 # edges between atoms derived from covalent bonds (source_n, target_n)
    edge_attr = []                                                                                  # bond features (e.g. bond type, ring-membership, …)
    for atom in mol.get_atoms():
        # getBonds()
        for bond in atom.GetBonds():
            edge_attr.append(get_bond_features(bond, bondfeatures))
            edge_index.append([atom.GetIdx(), bond.GetOtherAtomIdx(atom.GetIdx())])                 # (source_n, target_n)
            
        x.append(get_atom_features(atom, atomfeatures))

    data['x'] = torch.stack(x, dim=0)
    data['edge_index'] = torch.tensor(edge_index, dtype=torch.long).t().contiguous()                # edge connectivity
    # take care if edge_index is empty
    if len(edge_attr) == 0:
        edge_attr = torch.zeros(
            (data['edge_index'].size(1), len(bondfeatures)), dtype=torch.float32)                   # bond features  
    data['edge_attr'] = torch.stack(edge_attr, dim=0)                                               # edge features
    data['z'] = data['x'][:, 1].long()                                                              # atomic numbers
    data['idx'] = torch.tensor([idx], dtype=torch.long)                                             # molecule index

    data['complete_edge_index'] = torch.tensor([[i, j] for i in range(data['x'].size(0)) \
        for j in range(data['x'].size(0)) if i != j], dtype=torch.long).t().contiguous()            # complete graph connectivity

    data['y'] = target

    return Data(**data)