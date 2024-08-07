
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