import torch
from typing import Tuple, List
from rhea.chem.molecule import Molecule
from rhea.chem.structure import Structure

def protein_to_point_cloud(protein:Structure, features:Tuple[str|List[str]]) -> torch.Tensor:
    """
    The representation of a protein is a point cloud i.e. (X,Z) where
    Z belong to R^m*d is a matrix of m atoms with d features each and
    X belong to R^m*3 is a matrix of m 3D atoms coordinates.
    
    Parameters:
        protein (Structure): The protein.
        features (Tuple[str|List[str]]): The features to consider.
        
    Returns:
        torch.Tensor: The point cloud.
    """
    if isinstance(features[0], str):
        features = [features]
    
    coordinates = protein.get_coordinates()
    
    raise NotImplementedError("The protein_to_point_cloud function is not implemented yet.")

def molecule_to_point_cloud(mol:Molecule, features:Tuple[str|List[str]]) -> torch.Tensor:
    """
    The representation of a molecule is a point cloud i.e. (X,Z) where 
    Z belong to R^m*d is a matrix of m atoms with d features each and 
    X belong to R^m*3 is a matrix of m 3D atoms coordinates.

    Parameters:
        mol (Molecule): The molecule.
        features (Tuple[str|List[str]]): The features to consider.
    
    Returns:
        torch.Tensor: The point cloud.
    """
    if isinstance(features[0], str):
        features = [features]
    
    coordinates = mol.get_coordinates()

    feat = []
    for atom in mol.get_atoms():
        for feature in features:
            feat.append(atom.get_property(feature))
    
    return torch.cat([coordinates, torch.stack(feat, dim=1)], dim=1)

def molecule_to_Data(mol: Molecule, features: Tuple[str|List[str]]) -> dict:
    """
    Converts a molecule into a torch geometric `Data` object.
    
    The structure of the `Data` object is as follows:
    {
        'x': torch.Tensor,                   # node features  (e.g. formal charge, membership to aromatic rings, chirality, …)
        'edge_index': torch.Tensor,          # edges between atoms derived from covalent bonds (source_n, target_n)
        'edge_attr': torch.Tensor,           # bond features (e.g. bond type, ring-membership, …)
        'y': torch.Tensor,                   # target
        'pos': torch.Tensor,                 # atomic coordinates
        'idx': torch.Tensor,                 # node indices
        'name': str,                         # molecule name
        'z': torch.Tensor,                   # atomic numbers
        'complete_edge_index': torch.Tensor  # complete graph connectivity
    }
    """
    data = dict()
    pc = molecule_to_point_cloud(mol, features)
    data['x'] = pc[:, 3:]
    data['pos'] = pc[:, :3]


