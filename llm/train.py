import os
import sys
import yaml
import torch
import argparse
import torch.nn as nn
from tqdm import tqdm
from typing import Tuple
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter  # For TensorBoard
from logger import model_logger
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

def calc_loss_loader(loader: DataLoader, model: nn.Module, device: torch.device, eval_iter: int = None) -> float:
    total_loss = 0.0

    if len(loader) == 0:
        return float("nan")
    elif eval_iter is None:
        eval_iter = len(loader)
    else:
        eval_iter = min(eval_iter, len(loader))

    for i, (input_batch, target_batch) in enumerate(loader):
        loss = calc_loss_batch(input_batch, target_batch, model, device)
        total_loss += loss.item()
        if i == eval_iter:
            break
    return total_loss / eval_iter

def evaluate_model(
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        eval_iter: int = 100
        ) -> Tuple[float, float]:
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, eval_iter)
    model.train()
    return train_loss, val_loss

def calc_loss_batch(
        input_batch: torch.Tensor, 
        target_batch: torch.Tensor, 
        model: nn.Module, 
        device: torch.device) -> torch.Tensor:
    
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch) # tuple of (logits, hidden_states)
    loss = nn.functional.cross_entropy(logits[0].flatten(0, 1), target_batch.flatten())
    return loss
    

def load_config(path: str) -> dict:
    with open(path, "r") as file:
        config = yaml.safe_load(file)
    return config

def load_data(path: str) -> str:
    with open(path, "r") as file:
        data = file.read()
    return data

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
        if line == "<LIGAND>":
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

def train(
        config: dict,
        data: str,
        model_path: str,
        seed: int = 1234,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ):
    torch.random.manual_seed(seed=seed)
    
    if data is None or len(data) == 0:
        raise ValueError("The training data is empty.")
    
    vocab_size = len(set(data.split()))
    msg = f"Vocabulary size: {vocab_size}"
    model_logger.info(msg)

    # tokenizer
    tokenizer = LigandTokenizer()
    tokenizer.build_vocab([data])
    # save tokenizer
    tokenizer.save("tokenizer.json")

    #sys.exit("Exiting...")

    # split data
    train_data, val_data = split_data(data)

    # data loader
    train_loader = create_data_loader(
        input=train_data, 
        batch_size=config["batch_size"], 
        max_tokens=config["max_tokens"], 
        stride=config["stride"],
        tokenizer=tokenizer
    )
    val_loader = create_data_loader(
        input=val_data, 
        batch_size=config["batch_size"], 
        max_tokens=config["max_tokens"], 
        stride=config["stride"],
        tokenizer=tokenizer
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

    # log number of parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_logger.info(f"Number of parameters: {num_params}")

    # lr, optimizer, scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=float(config["lr"]), 
        weight_decay=config["weight_decay"]
    )    

    # Setup TensorBoard writer
    writer = SummaryWriter()

    # TRAINING
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen = 0
    global_step = -1     # global step counter
    eval_freq = config["eval_steps"]
    eval_iter = config["eval_iter"]

    for epoch in range(config["epochs"]):
        model.train()

        # Wrap train_loader with tqdm for progress display
        for input_batch, target_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}", leave=True):
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()   # number of elements in the input batch
            global_step += 1

            # Log the training loss to TensorBoard
            writer.add_scalar("Train/Loss", loss.item(), global_step)

            # optional evaluation
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model=model, 
                    train_loader=train_loader,
                    val_loader=val_loader,
                    device=device,
                    eval_iter=eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)

                # Log losses to TensorBoard
                writer.add_scalar("Train/Eval_Loss", train_loss, global_step)
                writer.add_scalar("Val/Eval_Loss", val_loss, global_step)

                msg = f"Epoch: {epoch}, Global step: {global_step}, Train loss: {train_loss}, Val loss: {val_loss}"
                model_logger.info(msg)

    # Close TensorBoard writer
    writer.close()

    # save model
    torch.save(model, model_path)
    
    return train_losses, val_losses, track_tokens_seen

def main():
    args = parse_args()
    config = load_config(args.config)
    decoder_cfg = config["DecoderTransformer"]

    if args.train_data is not None:
        data = load_data(args.train_data)
    else:
        raise ValueError("The training data is not provided.")

    train(
        decoder_cfg, 
        data,
        args.output
        )

if __name__ == '__main__':
    main()

#  cmd to init tensorboard
# tensorboard --logdir=runs