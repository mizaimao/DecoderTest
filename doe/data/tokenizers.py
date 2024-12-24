from typing import Callable, Dict, List, Tuple

import tiktoken


class CharTokenizer:
    def __init__(self, file_path: str):
        # Load the file contents.
        self.text: str
        with open(file_path, "r", encoding="utf-8") as f:
            self.text = f.read()
        # Create mapping info.
        chars: List[str] = sorted(list(set(self.text)))
        self.stoi: Dict[str, int] = {ch: i for i, ch in enumerate(chars)}
        self.itos: Dict[int, str] = {i: ch for i, ch in enumerate(chars)}
        self.vocabulary_size: int = len(chars)

    # Convert string into a list of integers.
    def encode(self, s: str) -> List[int]:
        if s is None:
            s = self.text
        return [self.stoi[c] for c in s]

    # Convert output logits back to a string.
    def decode(self, numbers: List[int]) -> str:
        return "".join([self.itos[i] for i in numbers])


class TikTokenWrapper:
    def __init__(self, file_path: str):
        # Load the file contents.
        self.text: str
        with open(file_path, "r", encoding="utf-8") as f:
            self.text = f.read()

        self.enc = tiktoken.get_encoding("gpt2")
        self.vocabulary_size: int = self.enc.max_token_value + 1
        
    def encode(self, s: str) -> List[int]:
        if s is None:
            s = self.text
        return self.enc.encode(s)
        
    def decode(self, numbers: List[int]) -> str:
        return self.enc.decode(numbers)

def get_tokenizer(
        file_path: str,
        name: str = "simple",
    ):

    if name == "simple":
        return CharTokenizer(
            file_path=file_path
        )

    elif name == "tiktoken":
        return TikTokenWrapper(
            file_path=file_path
        )

    else:
        raise NotImplementedError
