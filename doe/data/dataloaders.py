#from abc import ABC, abstractmethod
from typing import Dict, List, Tuple
from pathlib import Path
import torch

class TextLoader:

    def __init__(
        self,
        step_size: int,
        batch_size: int,
        file_path: str = "/home/frank/Projects/DecoderTest/text.txt",
        train_val_split: float = 0.8,
    ):
        # Sanity checks.
        if not Path(file_path).is_file():
            raise FileNotFoundError
        
        self.step_size: int = step_size
        self.batch_size: int = batch_size
        
        # Load the file contents.
        self.text: str
        with open(file_path, "r", encoding="utf-8") as f:
            self.text = f.read()
        # Create mapping info.
        chars: List[str] = sorted(list(set(self.text)))
        self.stoi: Dict[str, int] = {ch: i for i,ch in enumerate(chars) }
        self.itos: Dict[int, str] = {i: ch for i,ch in enumerate(chars) }
        self.vocabulary_size: int = len(chars)

        # Get split subsets.    
        self.train_data, self.valid_data = self.split_data(train_val_split)  

        
    
    def get_vocabulary_size(self):
        return self.vocabulary_size
    
    # Convert string into a list of integers.
    def encode(self, s: str) -> List[int]:
        return [self.stoi[c] for c in s]
        
    # Convert output logits back to a string.
    def decode(self, numbers: List[int]) -> str:
        return ''.join([self.itos[i] for i in numbers])
    
    def split_data(self, split: float) -> Tuple[torch.Tensor, torch.Tensor]:
        # Create a tensor to hold the encoded input.
        data: torch.Tensor = torch.tensor(self.encode(self.text), dtype=torch.long)
        print("Data loaded with shape and type:", data.shape, data.dtype)

        # Now separate the data into Train and Valid.
        train_data, valid_data = data[:int(len(data) * split)], data[int(len(data) * split):]
        print("Train length:", len(train_data), "Valid length:", len(valid_data))
        return train_data, valid_data
    
    def get_data(self, mode: str = "train") -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate a small batch of data. Both input and target."""
        if mode == "train":
            data: torch.Tensor = self.train_data
        elif mode == "valid":
            data = self.valid_data
        else:
            raise NotImplementedError
            
        indices: torch.Tensor = torch.randint(len(data) - self.step_size, (self.batch_size, ))
        x: torch.Tensor = torch.stack([data[i: i + self.step_size] for i in indices])
        y: torch.Tensor = torch.stack([data[i + 1: i + self.step_size + 1] for i in indices])
        return x, y