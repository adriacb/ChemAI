import os
import sys
import yaml
import torch
import torch.nn as nn

import argparse
from logger import model_logger
from typing import Tuple
from tokenizer import LigandTokenizer
from preprocessing.loader import create_data_loader

PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(PATH)
from llm.models.transformer import DecoderTransformer

CONFIG_TRAIN_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "config", "train.yml")
)

def parse_args():
    parser = argparse.ArgumentParser(description="Train the model.")
    parser.add_argument(
        "--config",
        type=str,
        default=CONFIG_TRAIN_PATH,
        help="Path to the configuration (yaml) file.",
    )
    parser.add_argument(
        "--train_data",
        type=str,
        default=None,
        help="Path to the training data.",
    )
    # output model
    parser.add_argument(
        "--output",
        type=str,
        default="model.pth",
        help="Path to the output model.",
    )
    args = parser.parse_args()
    return args

def load_data(path: str) -> str:
    with open(path, "r") as file:
        data = file.read()
    return data

def split_data(data: str, ratio: float = 0.8) -> Tuple[str, str]:
    """
    Split the data into training and validation sets.
    following this format:
    ```
    <LIGAND>
    smiles
    <XYZ>
    x y z
    <eos>
    ```
    """
    lines = data.strip().split("\n")
    ligands = []
    ligand = []
    for line in lines:
        if line == "<LIGAND>":
            ligand = []
        elif line == "<XYZ>":
            ligands.append(ligand)
        else:
            ligand.append(line)
    n_train = int(ratio * len(ligands))
    train_data = "\n".join(["\n".join(ligand) for ligand in ligands[:n_train]])
    val_data = "\n".join(["\n".join(ligand) for ligand in ligands[n_train:]])

    return train_data, val_data

def train(
        config: dict,
        data: str,
        seed: int = 1234,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ):
    torch.random.manual_seed(seed=seed)
    
    if data is None or len(data) == 0:
        raise ValueError("The training data is empty.")
    
    vocab_size = len(set(data.split()))
    msg = f"Vocabulary size: {vocab_size}"
    model_logger.info(msg)

    # split data
    train_data, val_data = split_data(data)

    # data loader
    train_loader = create_data_loader(
        input=train_data, 
        batch_size=config["batch_size"], 
        max_tokens=config["max_tokens"], 
        stride=config["stride"]
    )
    val_loader = create_data_loader(
        input=val_data, 
        batch_size=config["batch_size"], 
        max_tokens=config["max_tokens"], 
        stride=config["stride"]
    )
    len_train, len_val = len(train_loader), len(val_loader)
    model_logger.info(f"Number of training batches: {len_train}")
    model_logger.info(f"Number of validation batches: {len_val}")

    # model
    model = DecoderTransformer(
            vocab_size=vocab_size,                                      # Size of the vocabulary
            n_layers=config["n_layers"],                                # Number of layers in the transformer
            embedding_dim=config["embedding_dim"],                      # Dimension of the embeddings
            num_heads=config["num_heads"],                              # Number of heads in the multihead attention
            context_size=config["context_size"],                        # Size of the context window
            dropout=config["dropout"],                                  # Dropout probability
            )
    model.to(device=device)
    # lr, optimizer, scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=float(config["lr"]), 
        weight_decay=config["weight_decay"]
    )    


def load_config(path: str) -> dict:
    with open(path, "r") as file:
        config = yaml.safe_load(file)
    return config

def main():
    args = parse_args()
    config = load_config(args.config)
    decoder_cfg = config["DecoderTransformer"]

    if args.train_data is not None:
        data = load_data(args.train_data)
    else:
        raise ValueError("The training data is not provided.")

    train(decoder_cfg, data)

if __name__ == '__main__':
    main()