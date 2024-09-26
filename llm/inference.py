import sys
import argparse
import torch
from torch import nn
from torch.nn import functional as F
from typing import List
from tokenizer import LBPETokenizer
sys.path.append('..')
from llm.models.transformer import DecoderTransformer

MODEL_PATH = "model.pth"

TEST_INPUT = """<LIGAND>
Cn1c(=O)c2c(ncn2C)n(C)c1=O
<XYZ>"""

def format_output(output: List[str]) -> str:
    """Format the output list of tokens into a string."""
    smiles = ""
    xyz_data = []
    processing_xyz = False

    # Temporary storage for combining split coordinates
    temp_coordinates = []
    current_atom = None
    current_coord = ""

    coor_counter = 0
    i = 0
    for item in output:
        if item == '<LIGAND>':
            continue
        elif item == '<XYZ>':
            processing_xyz = True
            continue
        elif item == '<eos>':
            break

        if not processing_xyz:
            # Building SMILES string
            smiles += item
        else:
            if item.isalpha(): # Atom type
                current_atom = item
            else:
                # coordinates section
                current_coord += item
                i+=1
                
                if i == 2:
                    temp_coordinates.append(current_coord)
                    current_coord = ""
                    coor_counter += 1
                    i = 0
                if coor_counter == 3:
                    xyz_data.append(current_atom + " " + " ".join(temp_coordinates))
                    temp_coordinates = []
                    current_coord = ""
                    coor_counter = 0
                    i = 0
                                                           
    # Combine all parts into the desired output format
    output = f"<LIGAND>\n{smiles}\n<XYZ>\n" + "\n".join(xyz_data) + "\n<eos>"
    return output



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference.")
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_PATH,
        help="Path to the model.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=TEST_INPUT,
        help="Initial input sequence (as text or token indices).",
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=100,
        help="Maximum length of the generated sequence.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature.",
    )
    args = parser.parse_args()
    return args


def generate(model: nn.Module, idx: torch.Tensor, max_len: int = 100, temperature: float = 1.0) -> torch.Tensor:
    """Generate a sequence of tokens using the given model."""
    
    for _ in range(max_len):

        idx_cond = idx[:, -model.context_size:]  # (B, T)
        # get the predictions for the next token
        logits, _ = model.forward(idx_cond)
        # focus only on the last time step
        logits = logits[:, -1, :]               # becomes (B, C)
        # apply temperature
        logits = logits / max(temperature, 1e-5)           # used to control the randomness of the sampling
        # apply softmax
        probs = F.softmax(logits, dim=-1)       # (B, C)
        # sample the next token
        next_token = torch.multinomial(probs, num_samples=1) # (B, 1)
        # concatenate the new token
        idx = torch.cat([idx, next_token], dim=-1)
    return idx

def cached_generate(
        model: nn.Module, 
        idx: torch.Tensor, 
        max_len: int = 100, 
        temperature: float = 1.0) -> torch.Tensor:
    """Generate a sequence of tokens using the given model."""
    past_key_values = None

    for _ in range(max_len):
        # extract the last context size tokens
        idx_cond = idx[:, -model.context_size:]  # (B, T)

        # pass past_key_values to the model (None for the first iteration)
        outputs = model(input_ids=idx_cond, past_key_values=past_key_values)

        #extrect the logits
        logits = outputs.logits
        past_key_values = outputs.past_key_values

        # focus only on the last time step
        logits = logits[:, -1, :]               # becomes (B, C)
        # apply temperature
        logits = logits / max(temperature, 1e-5)           # used to control the randomness of the sampling
        # apply softmax
        probs = F.softmax(logits, dim=-1)       # (B, C)
        # sample the next token
        next_token = torch.multinomial(probs, num_samples=1)
        # concatenate the new token
        idx = torch.cat([idx, next_token], dim=-1)
    return idx

def resilient_generate(model: nn.Module, *args, **kwargs):
    oom:bool = False

    try:
        return generate(model, *args, **kwargs)
    except torch.cuda.OutOfMemoryError as e:
        print(e)
        oom = True
    if oom:
        torch.cuda.empty_cache()
        # change cache implementation: TODO
        return generate(model, *args, **kwargs)

def main():
    args = parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # load the model
    model = torch.load(args.model)
    model.to(device)
    model.eval()
    
    # load the tokenizer
    tokenizer = LBPETokenizer()
    tokenizer.load("LBPE.model")
    print(tokenizer.get_vocab_size())
    print(type(list(tokenizer.vocab.keys())[0]))


    if args.input:
        #tokens = tokenizer.tokenize(args.input)
        idx = tokenizer.encode(args.input)
        # check if any is None
        if None in idx:
            raise ValueError("Some tokens are not in the vocabulary.")
        
        idx = torch.tensor(idx).unsqueeze(0).to(device)

    else:
        raise ValueError("Please provide an input sequence.")
    

    ## TRAIN MORE TO GET BETTER RESULTS
    generated_idx = generate(
        model,
        idx,
        args.max_len,
        args.temperature
    )
    # generated_idx to list
    generated_idx = generated_idx.detach().cpu().numpy()[0]
    print(generated_idx)
    generated_text = tokenizer.decode(ids=generated_idx)
    print(generated_text)
    xyz_block = format_output(generated_text)
    print(xyz_block)

if __name__ == "__main__":
    main()


# kv cache:
# https://gist.github.com/ArthurZucker/af34221def212259b43d55a2811d2dbb

# continuous batching:
# https://www.anyscale.com/blog/continuous-batching-llm-inference

# Quantization:
