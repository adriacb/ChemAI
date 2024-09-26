import os
import sys
import argparse
from typing import Union
from logger import model_logger
from tokenizer import BaseTokenizer, GPT4Tokenizer
from data_utils import load_config, load_data
PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(PATH)


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

    

def train(
        config: dict,
        data: str,
        model_path: str=Union[str|None],
        ):
    
    if data is None or len(data) == 0:
        raise ValueError("The training data is empty.")

    model_logger.info("Tokenizer initialization and training.")
    tokenizer = GPT4Tokenizer(
        #vocab_size=config["tokenizer"]["vocab_size"]
    )
    tokenizer.register_special_tokens(
            ["<LIG>","<XYZ>","<MOL>","<FRAG>","<PROT>",
                "<POCKET>","<SURF>","<mask>","<bos>","<eos>","<pad>"]

        )
    tokenizer.train(data)

    # vocab size corresponds to the number of unique tokens in the training data
    vocab_size = tokenizer.get_vocab_size()             # len(set(data.split()))
    msg = f"Vocabulary size: {vocab_size}"
    model_logger.info(msg)
    #sys.exit("Exiting...")





def main():

    args = parse_args()
    config = load_config(args.config)
    
    if args.train_data is not None:
        data = load_data(args.train_data)
    else:
        raise ValueError("The training data is not provided.")
    
    train(
        config=config,
        data=data
    )
    


if __name__ == '__main__':
    main()

#  cmd to init tensorboard
# tensorboard --logdir=runs