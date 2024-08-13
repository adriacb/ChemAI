
import re
from transformers import AutoTokenizer
from pydantic import BaseModel, Field
from typing import List, Dict

# https://github.com/MTxSouza/MediumArticleGenerator/blob/main/model/tokenizer.py

class Tokenizer:
    SOS = "<s>"
    EOS = "<eos>"
    UNK = "<unk>"
    SOA = "<soa>"

    special_tokens_ = [SOS, EOS, UNK, SOA]


    def __init__(self, vocab: Dict[str, int]):
        self.vocab = vocab
        self.vocab_encode = {str(k): int(v) for k, v in vocab.items()}
        self.vocab_decode = {int(v): str(k) for k, v in vocab.items()}

    @staticmethod
    def create_vocab(corpus: str) -> Dict[str, int]:
        """
        Create a vocabulary from the given corpus.

        Args:
        - corpus: str

        Returns:
        - vocab: Dict[str, int]
        """
        vocab = {
            token: index
            for index, token in enumerate(sorted(set(corpus.split())))
        }

        # Special tokens for the transformer model
        vocab["<unk>"] = len(vocab)
        #vocab["<s>"] = len(vocab)
        #vocab["<eos>"] = len(vocab)
        
        return vocab
    
    def encode(self, text: str) -> List[int]:
        """
        Encode the given indices.

        Args:
        - indices: List[int]

        Returns:
        - tokens: List[str]
        """
        return [self.vocab_decode.get(char, "<unk>") for char in text]
    
    def decode(self, indices: List[int]) -> str:
        """
        Decode the given tokens.

        Args:
        - tokens: List[str]

        Returns:
        - text: str
        """
        return "".join([self.vocab_encode.get(char, "<unk>") for char in indices])

    def __len__(self) -> int:
        return len(self.vocab)

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize the given text.

        Args:
        - text: str

        Returns:
        - tokens: List[str]
        """
        taged_text = self.SOS + text + self.EOS + self.SOA

class LigandTokenizer(BaseModel):
    """
    Tokenizer for ligand SMILES and atom coordinates.

    ```
    input_text = '''
    <LIGAND>
    Cn1c(=O)c2c(ncn2C)n(C)c1=O
    <XYZ>
    C 3.0151 -1.4072 0.0796
    N 1.5857 -1.4661 -0.0918
    <eos>
    '''

    tokenizer = LigandTokenizer(prompt=input_text)
    tokens = tokenizer.tokenize()
    print(tokens)
    ```

    """
    # Maps to convert tokens to indices and vice versa
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
        parts = line.split()
        tokens = []
        for part in parts:
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
        return tokens
    
    def tokenize(self, text: str) -> List[str]:
        """
        Main method to tokenize the entire input text based on defined rules.
        """
        tokens = []
        lines = text.strip().split('\n')
        
        processing_smiles = False
        
        for line in lines:
            line = line.strip()
            if line in ['<LIGAND>', '<XYZ>', '<eos>']:
                tokens.append(line)
                if line == '<LIGAND>':
                    processing_smiles = True
                elif line == '<XYZ>':
                    processing_smiles = False
            elif processing_smiles:
                tokens.extend(self.tokenize_ligand(line))
                processing_smiles = False  # End processing SMILES after first line
            elif line[0].isupper():  # Likely an atom coordinate line
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
        unique_tokens = set(all_tokens)
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