import os
import sys
import yaml
import torch
import torch.nn as nn
import argparse
from logger import model_logger
from typing import List, Dict
from tokenizer import LigandTokenizer

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

def train(
        config: dict,
        data: str
        ):
    torch.random.manual_seed(seed=1234)
    
    if data is None or len(data) == 0:
        raise ValueError("The training data is empty.")
    
    vocab_size = len(set(data.split()))
    msg = f"Vocabulary size: {vocab_size}"
    model_logger.warning(msg)

    # tokenizer
    tokenizer = LigandTokenizer()
    tokenizer.build_vocab([data])

    tokens = tokenizer.tokenize(data)
    encoded_data = tokenizer.encode(tokens)

    msg = f"Number of tokens: {len(tokens)}"
    model_logger.warning(msg)

    # model
    model = DecoderTransformer(
            vocab_size=vocab_size,                                      # Size of the vocabulary
            n_layers=config["n_layers"],                                # Number of layers in the transformer
            embedding_dim=config["embedding_dim"],                      # Dimension of the embeddings
            num_heads=config["num_heads"],                              # Number of heads in the multihead attention
            context_size=config["context_size"],                        # Size of the context window
            dropout=config["dropout"],                                  # Dropout probability
            )
    model.to(device=config["device"])
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