import os
import torch
import json
import regex as re
import unicodedata
from transformers import AutoTokenizer
from pydantic import BaseModel, Field
from typing import List, Dict, Tuple, Union
from reg import *
# https://github.com/MTxSouza/MediumArticleGenerator/blob/main/model/tokenizer.py

def get_stats(ids: List[int], counts: Dict[int, int] = None) -> Dict[Tuple[int, int], int]:
    """Given a list of integers, return a dictionary of counts for each pair of integers.
    E.g [1, 2, 3, 1, 2, 1] -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}
    
    Args:
        ids (List[int]): List of integers.
        counts (Dict[int, int], optional): Dictionary of counts for each pair of integers. Defaults to None.
    
    Returns:
        Dict[Tuple[int, int], int]: Dictionary of counts for each pair of integers.
    """
    counts = counts or {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids: List[int], pair: Tuple[int, int], idx: int) -> List[int]:
    """Merge a pair of integers in a list of integers.
    ```
    E.g [1, 2, 3, 1, 2, 1], (1, 2), 0 -> [0, 0, 3, 0, 0, 0]
    ```
    Args:
        ids (List[int]): List of integers.
        pair (Tuple[int, int]): Pair of integers to merge.
        idx (int): Index of the new integer.
    
    Returns:
        List[int]: List of integers with the merged pair.
    """
    new_ids = []
    i = 0
    while i < len(ids):
        if ids[i] == pair[0] and i < len(ids) - 1 and ids[i + 1] == pair[1]:
            new_ids.append(idx)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids

def render_token(token: bytes) -> str:
    """Render a token as a string.
    Args:
        token (bytes): Token as bytes.
    Returns:
        str: Token as a string.
    """
    s = token.decode('utf-8', errors='replace')
    chars = []
    for ch in s:
        if unicodedata.category(ch).startswith('C'):
            chars.append(ch)
        else:
            chars.append(f"\\u{ord(ch):04x}")
    return "".join(chars)

 
    
class Tokenizer:
    """Base class for Tokenizers"""

    def __init__(self):
        # default: vocab size of 256 (all bytes), no merges, no patterns
        self.merges = {} # (int, int) -> int
        self.pattern = "" # str
        self.special_tokens = {} # str -> int, e.g. {'<|endoftext|>': 100257}
        self.vocab = self._build_vocab() # int -> bytes

    def train(self, text: str, vocab_size: int = 256, verbose: bool = False):
        # Tokenizer can train a vocabulary of size vocab_size from text
        raise NotImplementedError

    def encode(self, text: str):
        # Tokenizer can encode a string into a list of integers
        raise NotImplementedError

    def decode(self, ids: List[int]):
        # Tokenizer can decode a list of integers into a string
        raise NotImplementedError

    def _build_vocab(self):
        # vocab is simply and deterministically derived from merges
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")
        return vocab

    def get_tokens(self, text: str) -> List[str]:
        """
        Given a text, return the list of tokens in their textual representation.
        
        Args:
            text (str): The input text to tokenize.

        Returns:
            List[str]: List of tokens as text.
        """
        # Encode the text into token IDs
        token_ids = self.encode(text)
        print(set(token_ids))
        # Convert each token ID back into its string representation
        tokens = [self.vocab[token_id].decode('utf-8', errors='replace') for token_id in token_ids]
        
        return tokens        

    def save(self, file_prefix: str):
        """
        Saves two files: file_prefix.vocab and file_prefix.model
        This is inspired (but not equivalent to!) sentencepiece's model saving:
        - model file is the critical one, intended for load()
        - vocab file is just a pretty printed version for human inspection only
        """
        # write the model: to be used in load() later
        model_file = file_prefix + ".model"
        with open(model_file, 'w') as f:
            # write the version, pattern and merges, that's all that's needed
            f.write("minbpe v1\n")
            f.write(f"{self.pattern}\n")
            # write the special tokens, first the number of them, then each one
            f.write(f"{len(self.special_tokens)}\n")
            for special, idx in self.special_tokens.items():
                f.write(f"{special} {idx}\n")
            # the merges dict
            for idx1, idx2 in self.merges:
                f.write(f"{idx1} {idx2}\n")
        # write the vocab: for the human to look at
        vocab_file = file_prefix + ".vocab"
        inverted_merges = {idx: pair for pair, idx in self.merges.items()}
        with open(vocab_file, "w", encoding="utf-8") as f:
            for idx, token in self.vocab.items():
                # note: many tokens may be partial utf-8 sequences
                # and cannot be decoded into valid strings. Here we're using
                # errors='replace' to replace them with the replacement char �.
                # this also means that we couldn't possibly use .vocab in load()
                # because decoding in this way is a lossy operation!
                s = render_token(token)
                # find the children of this token, if any
                if idx in inverted_merges:
                    # if this token has children, render it nicely as a merge
                    idx0, idx1 = inverted_merges[idx]
                    s0 = render_token(self.vocab[idx0])
                    s1 = render_token(self.vocab[idx1])
                    f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
                else:
                    # otherwise this is leaf token, just print it
                    # (this should just be the first 256 tokens, the bytes)
                    f.write(f"[{s}] {idx}\n")

    def load(self, model_file: str):
        """Inverse of save() but only for the model file"""
        assert model_file.endswith(".model")
        # read the model file
        merges = {}
        special_tokens = {}
        idx = 256
        with open(model_file, 'r', encoding="utf-8") as f:
            # read the version
            version = f.readline().strip()
            assert version == "minbpe v1"
            # read the pattern
            self.pattern = f.readline().strip()
            # read the special tokens
            num_special = int(f.readline().strip())
            for _ in range(num_special):
                special, special_idx = f.readline().strip().split()
                special_tokens[special] = int(special_idx)
            # read the merges
            for line in f:
                idx1, idx2 = map(int, line.split())
                merges[(idx1, idx2)] = idx
                idx += 1
        self.merges = merges
        self.special_tokens = special_tokens
        self.vocab = self._build_vocab()


class LBPETokenizer(Tokenizer):
    def __init__(self, pattern: Union[str, None] = None):
        super().__init__()
        self.pattern: str = LIGAND_PAT if pattern is None else pattern
        self.compiled_pattern: re.Pattern = re.compile(self.pattern)
        self.special_tokens: Dict[str, int] = dict()
        self.inverse_special_tokens: Dict[int, str] = dict()
    
    def train(self, text: str, vocab_size: int = 256, verbose: bool = False):
        if vocab_size < 256:
            raise ValueError("Vocab size must be at least 256.")
        num_merges = vocab_size - 256
        text_chunks = re.findall(self.compiled_pattern, text)
        ids = [list(ch.encode('utf-8')) for ch in text_chunks]
        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for i in range(num_merges):
            stats = {}
            for chunk_ids in ids:
                # passing in stats will update it in place
                get_stats(chunk_ids, stats)
            pair = max(stats, key=stats.get)
            idx = 256 + i
            ids = [merge(chunk_ids, pair, idx) for chunk_ids in ids]
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")

        self.merges = merges
        self.vocab = vocab
    
    def register_special_tokens(self, special_tokens: Dict[str, int]):
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {idx: token for token, idx in special_tokens.items()}

        # Add special tokens to the vocab
        for token, idx in special_tokens.items():
            self.vocab[idx] = token.encode("utf-8")

    def decode(self, ids: List[int]) -> str:
        part_bytes = []
        for idx in ids:
            if idx in self.vocab:
                part_bytes.append(self.vocab[idx])
            elif idx in self.inverse_special_tokens:
                part_bytes.append(self.inverse_special_tokens[idx].encode("utf-8"))
            else:
                raise ValueError(f"invalid token id: {idx}")
        text_bytes = b"".join(part_bytes)
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    def _encode_chunk(self, text_bytes: bytes) -> List[int]:
        ids = list(text_bytes)
        while len(ids) >= 2:
            # find the pair with the lowest merge index
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            # subtle: if there are no more merges available, the key will
            # result in an inf for every single pair, and the min will be
            # just the first pair in the list, arbitrarily
            # we can detect this terminating case by a membership check
            if pair not in self.merges:
                break # nothing else can be merged anymore
            # otherwise let's merge the best pair (lowest merge index)
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids
    
    def encode_ordinary(self, text: str) -> List[int]:
        """Encoding that ignores any special tokens."""
        # split text into chunks of text by categories defined in regex pattern
        text_chunks = re.findall(self.compiled_pattern, text)
        # all chunks of text are encoded separately, then results are joined
        ids = []
        for chunk in text_chunks:
            chunk_bytes = chunk.encode("utf-8") # raw bytes
            chunk_ids = self._encode_chunk(chunk_bytes)
            ids.extend(chunk_ids)
        return ids
    
    def encode(self, text: str, allowed_special: str = "all") -> List[int]:
        """
        Unlike encode_ordinary, this function handles special tokens.
        allowed_special: can be "all"|"none"|"none_raise" or a custom set of special tokens
        if none_raise, then an error is raised if any special token is encountered in text
        this is the default tiktoken behavior right now as well
        any other behavior is either annoying, or a major footgun
        """
        # decode the user desire w.r.t. handling of special tokens
        special = None
        if allowed_special == "all":
            special = self.special_tokens
        elif allowed_special == "none":
            special = {}
        elif allowed_special == "none_raise":
            special = {}
            assert all(token not in text for token in self.special_tokens)
        elif isinstance(allowed_special, set):
            special = {k: v for k, v in self.special_tokens.items() if k in allowed_special}
        else:
            raise ValueError(f"allowed_special={allowed_special} not understood")
        if not special:
            # shortcut: if no special tokens, just use the ordinary encoding
            return self.encode_ordinary(text)
        # otherwise, we have to be careful with potential special tokens in text
        # we handle special tokens by splitting the text
        # based on the occurrence of any exact match with any of the special tokens
        # we can use re.split for this. note that surrounding the pattern with ()
        # makes it into a capturing group, so the special tokens will be included
        special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")"
        special_chunks = re.split(special_pattern, text)
        # now all the special characters are separated from the rest of the text
        # all chunks of text are encoded separately, then results are joined
        ids = []
        for part in special_chunks:
            if part in special:
                # this is a special token, encode it separately as a special case
                ids.append(special[part])
            else:
                # this is an ordinary sequence, encode it normally
                ids.extend(self.encode_ordinary(part))
        return ids  

class LigandTokenizer(BaseModel):
    """
    Tokenizer for ligand SMILES and atom coordinates.

    Example usage:
    ```
    input_text = '''
    <LIGAND>
    Cn1c(=O)c2c(ncn2C)n(C)c1=O
    <XYZ>
    C 3.0151 -1.4072 0.0796
    N 1.5857 -1.4661 -0.0918
    <eos>
    '''

    tokenizer = LigandTokenizer()
    tokens = tokenizer.tokenize(input_text)
    print(tokens)
    ```
    """
    token_to_id: Dict[str, int] = Field(default_factory=dict)
    id_to_token: Dict[int, str] = Field(default_factory=dict)
    
    def tokenize_ligand(self, ligand: str) -> List[str]:
        """
        Tokenizes the SMILES string where each character is a separate token.
        """
        return list(ligand)
    
    def tokenize_coordinates(self, line: str) -> List[str]:
        """
        Tokenizes the atom and its coordinates.
        Splits the atom label as a token and each part of the coordinate as separate tokens.
        """
        parts = re.split(r'(\s+)', line)  # Preserve spaces as tokens
        tokens = []
        for part in parts:
            if part.strip():  # Non-whitespace part
                if re.match(r'^[A-Z][a-z]?', part):  # Atom labels like C, N, O
                    tokens.append(part)
                else:
                    # Split number into integer and fractional parts
                    if '.' in part:
                        integer_part, fractional_part = part.split('.')
                        tokens.append(integer_part)
                        tokens.append(f".{fractional_part}")
                    else:
                        tokens.append(part)
            else:
                tokens.append(part)  # Add space as a token
        return tokens
    
    def tokenize(self, text: str) -> List[str]:
        """
        Main method to tokenize the entire input text based on defined rules.
        """
        tokens = []
        lines = re.split(r'(\n)', text)  # Preserve newlines as tokens
        
        processing_smiles = False
        
        for line in lines:
            if line == '\n':
                tokens.append(line)
                continue
            
            line = line.strip()
            if line in ['<LIGAND>', '<XYZ>', '<eos>']:
                tokens.append(line)
                if line == '<LIGAND>':
                    processing_smiles = True
                elif line == '<XYZ>':
                    processing_smiles = False
            elif processing_smiles:
                tokens.extend(self.tokenize_ligand(line))
                processing_smiles = False
            elif line and line[0].isupper():  # Likely an atom coordinate line
                tokens.extend(self.tokenize_coordinates(line))

        return tokens
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build a vocabulary from a list of input texts.
        """
        all_tokens = []
        for text in texts:
            tokens = self.tokenize(text)
            all_tokens.extend(tokens)
        unique_tokens = sorted(set(all_tokens))
        self.token_to_id = {token: idx for idx, token in enumerate(unique_tokens)}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}
    
    def encode(self, tokens: List[str]) -> List[int]:
        """
        Encode tokens into a list of integers based on the vocabulary.
        """
        if not self.token_to_id:
            raise ValueError("Vocabulary not built. Call `build_vocab` first.")
        return [self.token_to_id.get(token, self.token_to_id.get('<UNK>')) for token in tokens]
    
    def decode(self, indices: List[int]) -> List[str]:
        """
        Decode a list of integers back into tokens.
        """
        if not self.id_to_token:
            raise ValueError("Vocabulary not built. Call `build_vocab` first.")
        return [self.id_to_token.get(idx, '<UNK>') for idx in indices]
    
    def decode_tkn(self, idx: torch.Tensor) -> List[str]:
        """
        Decode a tensor of integers back into tokens.
        """
        if not self.id_to_token:
            raise ValueError("Vocabulary not built. Call `build_vocab` first.")
        
        if idx.dim() > 1:
            idx = idx.squeeze(0)
            return [self.id_to_token.get(int(idx[i].item()), '<UNK>') for i in range(idx.size(0))]
        else:
            return [self.id_to_token.get(int(idx[i].item()), '<UNK>') for i in range(idx.size(0))]
        
    def save(self, path: str) -> None:
        """
        Save the tokenizer to a JSON file.
        """
        # check if already exists
        if os.path.exists(path):
            print(f"Removing existing file: {path}")
            os.remove(path)
        
        with open(path, 'w') as f:
            json.dump({
                'token_to_id': self.token_to_id,
                'id_to_token': self.id_to_token
            }, f)

    @classmethod
    def load(cls, path: str) -> 'LigandTokenizer':
        """
        Load the tokenizer from a JSON file.
        """
        # usage LigandTokenizer.load("tokenizer.json")
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)

class GPTTokenizer:

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    EOA = tokenizer.eos_token

    def __init__(self):
        """
        Bert tokenizer class for converting text to numbers and vice versa.
        """
        self.tokenizer = self.tokenizer

    def __len__(self) -> int:
        """
        Return the size of vocabulary.

        Returns:
            int : The size of vocabulary.
        """
        return self.tokenizer.vocab_size

    @property
    def pad_index(self) -> int:
        """
        Get the index of padding token.

        Returns:
            int : The index of padding token.
        """
        return self.tokenizer.bos_token_id

    def get_vocab(self) -> Dict[str, int]:
        """
        Get the vocabulary of Tokenizer.

        Returns:
            Dict[str, int] : The vocabulary dictionary.
        """
        return self.tokenizer.get_vocab()

    def encode(self, text: str) -> List[int]:
        """
        Tokenize the input text into a list of tokens.

        Args:
            text (str) : The input text.
        
        Returns:
            List[int] : The list of token indices.
        """
        return self.tokenizer.encode(text=text, add_special_tokens=False)

    def decode(self, indices: List[int]) -> str:
        """
        Decode the list of indices into a text.

        Args:
            indices (List[int]) : The list of indices.

        Returns:
            str : The output text.
        """
        return self.tokenizer.decode(token_ids=indices, skip_special_tokens=False)

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize the input text into a list of tokens and add special tokens.

        Args:
            text (str) : The input text.

        Returns:
            List[str] : The list of tokens.
        """
        tagged_text = text + "\n\n"
        return self.encode(text=tagged_text)