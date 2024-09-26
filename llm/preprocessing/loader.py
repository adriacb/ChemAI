import sys
sys.path.append('..')
import torch
from typing import List, Tuple
from llm.tokenizer import LigandTokenizer
from torch.utils.data import DataLoader
from llm.logger import model_logger

class MoleculeDatasetLoader:
    def __init__(self, text: str, 
                 tokenizer: LigandTokenizer, 
                 max_tokens: int,
                 stride: int):
        """Constructor for the MoleculeDatasetLoader class.

        Args:
        - text: str, the text data
        - tokenizer: LigandTokenizer, the tokenizer
        - max_tokens: int, the maximum number of tokens
        - stride: int, the stride"""
        
        self.text = text
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens

        self.input_ids = []
        self.target_ids = []

        #tokens = self.tokenizer.tokenize(text)
        token_ids = self.tokenizer.encode(text)
        
        if None in token_ids:
            raise ValueError(
                "The token IDs contain None values. " \
                "Please check the tokenization process."
                )

        msg = f"Number of tokens: {len(token_ids)}"
        model_logger.info(msg)

        for i in range(0, len(token_ids) - max_tokens, stride):
            input_chunk = token_ids[i:i+max_tokens]           # input chunk of size max_tokens
            target_chunk = token_ids[i+1:i+max_tokens+1]      # target chunk of size max_tokens
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self) -> int:
        """Return the number of examples."""
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> List[torch.Tensor]:
        """Return the input and target tensors at the given index."""
        return self.input_ids[idx], self.target_ids[idx]
    

def create_data_loader(input: str, 
                       batch_size: int, 
                       max_tokens: int, 
                       stride: int,
                       shuffle: bool = True, 
                       drop_last: bool = True, 
                       num_workers: int = 0,
                       tokenizer: LigandTokenizer = None
                       ) -> DataLoader:
    """Create a DataLoader for the given input data.

    Args:
    - input: str, the input data
    - batch_size: int, the batch size
    - max_tokens: int, the maximum number of tokens
    - stride: int, the stride
    - shuffle: bool, whether to shuffle the data
    - drop_last: bool, whether to drop the last incomplete batch
    - num_workers: int, the number of workers

    Returns:
    - DataLoader: the DataLoader"""

    if len(input) == 0:
        raise ValueError("The input data is empty.")

    if tokenizer is None:
        raise ValueError("The tokenizer is not provided.")

    dataset = MoleculeDatasetLoader(
        input, tokenizer, max_tokens, stride)
    
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return data_loader

def split_data(data: str, ratio: float = 0.8) -> Tuple[str, str]:
    """
    Split the data into training and validation sets.
    The format should be:
    
    <LIGAND>
    smiles
    <XYZ>
    x y z
    <eos>
    """
    lines = data.strip().split("\n")
    ligands = []
    ligand = []
    for line in lines:
        if line == "<MOL>":
            if ligand:  # Don't forget to add the previous ligand if it exists
                ligands.append(ligand)
            ligand = [line]
        elif line == "<XYZ>":
            ligand.append(line)
        elif line == "<eos>":
            ligand.append(line)
            ligands.append(ligand)
            ligand = []
        else:
            ligand.append(line)

    # Handle the case where data does not end with <eos>
    if ligand:
        ligands.append(ligand)
    
    n_train = int(ratio * len(ligands))
    train_data = "\n".join(["\n".join(ligand) for ligand in ligands[:n_train]])
    val_data = "\n".join(["\n".join(ligand) for ligand in ligands[n_train:]])

    return train_data, val_data

def split_data2(data: str, ratio: float = 0.8) -> Tuple[str, str]:
    """
    Split the data into training and validation sets based on samples starting with <MOL>.
    Each sample starts with <MOL> and contains:
    
    <MOL>
    smiles
    <XYZ>
    x y z
    """
    lines = data.strip().split("\n")
    ligands = []
    ligand = []
    
    for line in lines:
        if line == "<MOL>":
            if ligand:  # If we encounter a new <MOL>, store the previous ligand
                ligands.append(ligand)
            ligand = [line]  # Start a new ligand entry
        else:
            ligand.append(line)
    
    # Add the last ligand (in case the data doesn't end with a <MOL>)
    if ligand:
        ligands.append(ligand)
    
    # Split into training and validation sets
    n_train = int(ratio * len(ligands))
    train_data = "\n".join(["\n".join(ligand) for ligand in ligands[:n_train]])
    val_data = "\n".join(["\n".join(ligand) for ligand in ligands[n_train:]])

    return train_data, val_data