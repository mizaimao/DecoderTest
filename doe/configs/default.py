from typing import Dict, Tuple, List
from dataclasses import dataclass


@dataclass
class DefaultConfig:
    seed: int = 421
    train_valid_split: int = 0.8

    batch_size: int = 24
    epochs: int = (
        20000  # Not exactly epochs by definitaion but more like an iteration number.
    )
    step_size: int = 512  # Max num characters to look back.
    embedding_dim: int = 256
    n_blocks: int = 16
    n_heads: int = 4
    lr: float = 3e-4
    dropout: float = 0.2
    preview_size: int = 200
    print_every: int = 500

    input_path: str = "/home/frank/Projects/DecoderTest/inputs/news20_concated.txt"

    