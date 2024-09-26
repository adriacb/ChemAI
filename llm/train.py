import os
import sys
import yaml
import torch
import argparse
import torch.nn as nn
from tqdm import tqdm
from typing import List, Tuple
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter  # For TensorBoard
# https://www.tensorflow.org/install/pip?hl=es-419#linux
from logger import model_logger
from tokenizer import LBPETokenizer#LigandTokenizer
from preprocessing.loader import create_data_loader, split_data2
PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(PATH)
from llm.models.transformer import DecoderTransformer
from llm.inference import generate

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

def plot_losses(train_losses: List[float], 
                val_losses: List[float], 
                epochs_seen: List[int], 
                tokens_seen: List[int]
                ) -> None:
    fig, ax1 = plt.subplots()
    #plot training and validation loss against epochs
    ax1.plot(epochs_seen, train_losses, label="Train loss", color="blue", linestyle="--")
    ax1.plot(epochs_seen, val_losses, label="Val loss", color="red", linestyle="--")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")

    # create a second x-axis for tokens seen
    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()
    plt.savefig("losses.png")
    
def calc_loss_loader(loader: 
                     DataLoader, 
                     model: nn.Module, 
                     device: torch.device, 
                     eval_iter: int = None) -> float:
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

def train(
        config: dict,
        data: str,
        model_path: str,
        seed: int = 123,
        print_samples: bool = True,
        print_freq: int = 1_000,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ):
    torch.manual_seed(seed=seed)
    torch.cuda.manual_seed(seed=seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    
    if data is None or len(data) == 0:
        raise ValueError("The training data is empty.")
    
    model_logger.info(f"Using device {device}")

    # tokenizer
    # tokenizer = LBPETokenizer()#LigandTokenizer()
    # tokenizer.build_vocab([data])
    # # save tokenizer
    # tokenizer.save("tokenizer.json")
    model_logger.info("Tokenizer initialization and training.")
    tokenizer = LBPETokenizer()
    tokenizer.train(data)
    tokenizer.register_special_tokens(
        {
            "<LIG>": 257,
            "<XYZ>": 258,
            "<MOL>": 259,
            "<FRAG>": 260,
            "<PROT>": 261,
            "<POCKET>": 262,
            "<SURF>": 267,
            "<mask>": 263,
            "<bos>": 264,
            "<eos>": 265,
            "<pad>": 266,
        }
    )

    # vocab size corresponds to the number of unique tokens in the training data
    vocab_size = tokenizer.get_vocab_size()             # len(set(data.split()))
    msg = f"Vocabulary size: {vocab_size}"
    model_logger.info(msg)
    #sys.exit("Exiting...")

    # split data
    train_data, val_data = split_data2(data)

    # data loader
    train_loader = create_data_loader(
        input=train_data, 
        batch_size=config["training"]["batch_size"], 
        max_tokens=config["tokenizer"]["max_tokens"], 
        stride=config["tokenizer"]["stride"],
        tokenizer=tokenizer
    )
    val_loader = create_data_loader(
        input=val_data, 
        batch_size=config["training"]["batch_size"], 
        max_tokens=config["tokenizer"]["max_tokens"], 
        stride=config["tokenizer"]["stride"],
        tokenizer=tokenizer
    )

    len_train, len_val = len(train_loader), len(val_loader)
    model_logger.info(f"Number of training batches: {len_train}")
    model_logger.info(f"Number of validation batches: {len_val}")
    
    # model
    model = DecoderTransformer(
            vocab_size=vocab_size,                                      # Size of the vocabulary
            n_layers=config["DecoderTransformer"]["n_layers"],                                # Number of layers in the transformer
            embedding_dim=config["DecoderTransformer"]["embedding_dim"],                      # Dimension of the embeddings
            num_heads=config["DecoderTransformer"]["num_heads"],                              # Number of heads in the multihead attention
            context_size=config["DecoderTransformer"]["context_size"],                        # Size of the context window
            dropout=config["DecoderTransformer"]["dropout"],
            activation=config["DecoderTransformer"]["activation"]
            )
    model.to(device=device)

    # store image summary of the model
    

    # log number of parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_logger.info(f"Number of parameters: {num_params}")

    # lr, optimizer, scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=float(config["training"]["lr"]), 
        weight_decay=config["training"]["weight_decay"]
    )    

    # Setup TensorBoard writer
    writer = SummaryWriter()

    # TRAINING
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen = 0
    global_step = -1     # global step counter
    eval_freq = config["training"]["eval_steps"]
    eval_iter = config["training"]["eval_iter"]

    for epoch in range(config["training"]["epochs"]):
        model.train()

        # Wrap train_loader with tqdm for progress display
        for input_batch, target_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['training']['epochs']}", leave=True):
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

            # logic to print samples
            if print_samples and global_step % print_freq == 0:
                generate_and_print_sample(
                    model=model, 
                    tokenizer=tokenizer, 
                    device=device, 
                    start_context=str(config["DecoderTransformer"]["start_context"])
                )
    # Close TensorBoard writer
    writer.close()

    # save model
    torch.save(model, model_path)
    
    return train_losses, val_losses, track_tokens_seen

def generate_and_print_sample(
        model: nn.Module,
        tokenizer: LBPETokenizer,
        device: torch.device,
        start_context: str
        ) -> None:
    model.eval()
    context_size = model.pos_embedding.weight.shape[0]
    tokens = tokenizer.tokenize(start_context)
    encoded = tokenizer.encode(tokens)
    encoded = torch.tensor(
        data=encoded,
        dtype=torch.long            # IMPORTANT: the data type should be long
        ).unsqueeze(0).to(device)
    
    if encoded.size(1) == 0:  # Check if the encoded tensor is empty after encoding
        print("Encoded tensor is empty. Please check the input or encoding process.")
        return

    with torch.no_grad():
        tokens_ids = generate(
            model=model, 
            idx=encoded, 
            max_len=context_size, 
            temperature=1.0)
        
        if tokens_ids is None or len(tokens_ids) == 0:
            print("Generation returned an empty token list. Please check the model and generation process.")
            return
        
        decoded_tokens = tokenizer.decode_tkn(tokens_ids)
        print("".join(decoded_tokens))
    model.train()



def main():

    args = parse_args()
    config = load_config(args.config)
    decoder_cfg = config

    if args.train_data is not None:
        data = load_data(args.train_data)
    else:
        raise ValueError("The training data is not provided.")
    
    # clean cache
    torch.cuda.empty_cache()

    train_losses, val_losses, track_tokens_seen = train(
        config=decoder_cfg, 
        data=data,
        model_path=args.output,
        seed=config["training"]["seed"],
        print_samples=config["training"]["print_samples"]
        )
    
    plot_losses(
        train_losses, 
        val_losses, 
        list(range(len(train_losses))), 
        track_tokens_seen)

if __name__ == '__main__':
    main()

#  cmd to init tensorboard
# tensorboard --logdir=runs