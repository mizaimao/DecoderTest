# from abc import ABC, abstractmethod
from typing import Dict, List, Tuple
from pathlib import Path
import torch

from doe.data.tokenizers import get_tokenizer
from doe.configs.default import DefaultConfig


def get_loader(name: str = "text", config: DefaultConfig = None):
    loader = None
    if name == "text":
        loader: TextLoader = TextLoader(
            step_size=config.step_size,
            batch_size=config.batch_size,
            file_path=config.input_path,
            train_val_split=config.train_valid_split,
        )

    else:
        raise NotImplementedError

    return loader


class TextLoader:

    def __init__(
        self,
        step_size: int,
        batch_size: int,
        file_path: str = "/home/frank/Projects/DecoderTest/inputs/text.txt",
        train_val_split: float = 0.8,
    ):
        # Sanity checks.
        if not Path(file_path).is_file():
            raise FileNotFoundError
        self.tokenizer = get_tokenizer(file_path=file_path, name="tiktoken")

        self.step_size: int = step_size
        self.batch_size: int = batch_size

        # Get split subsets.
        self.train_data, self.valid_data = self.split_data(train_val_split)

    def get_vocabulary_size(self):
        return self.tokenizer.vocabulary_size

    def decode(self, numbers: List[int]) -> str:
        return self.tokenizer.decode(numbers=numbers)

    def split_data(self, split: float) -> Tuple[torch.Tensor, torch.Tensor]:
        # Create a tensor to hold the encoded input.
        data: torch.Tensor = torch.tensor(self.tokenizer.encode(None), dtype=torch.long)
        print("Data loaded with shape and type:", data.shape, data.dtype)

        # Now separate the data into Train and Valid.
        train_data, valid_data = (
            data[: int(len(data) * split)],
            data[int(len(data) * split) :],
        )
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

        indices: torch.Tensor = torch.randint(
            len(data) - self.step_size, (self.batch_size,)
        )
        x: torch.Tensor = torch.stack([data[i : i + self.step_size] for i in indices])
        y: torch.Tensor = torch.stack(
            [data[i + 1 : i + self.step_size + 1] for i in indices]
        )
        return x, y
