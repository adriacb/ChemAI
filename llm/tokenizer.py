
from transformers import AutoTokenizer

from typing import Dict, List
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

class StructTokenizer:
    SOS = "<s>"
    EOS = "<eos>"
    UNK = "<unk>"
    PROT = "<PROTEIN>"
    LIG = "<LIGAND>"
    SOA = "<$$$$>"

    special_tokens_ = [SOS, EOS, UNK, PROT, LIG]


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
        # TODO: Implement the create_vocab method
        raise NotImplementedError
    
    def encode(self, text: str) -> List[int]:
        """
        Encode the given indices.

        Args:
        - indices: List[int]

        Returns:
        - tokens: List[str]
        """
        # TODO: Implement the encode method
        raise NotImplementedError
    
    def decode(self, indices: List[int]) -> str:
        """
        Decode the given tokens.

        Args:
        - tokens: List[str]

        Returns:
        - text: str
        """
        # TODO: Implement the decode method
        raise NotImplementedError

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