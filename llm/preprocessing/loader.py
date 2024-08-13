import sys
sys.path.append('..')
from tokenizer import LigandTokenizer
from torch.utils.data import DataLoader

class MoleculeDatasetLoader:
    def __init__(self, text: str, 
                 tokenizer: LigandTokenizer, 
                 max_tokens: int = 100):
        self.text = text
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens

        self.input_ids = []
        self.target_ids = []