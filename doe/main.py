from typing import *
import torch.nn as nn
from torch.nn import functional as F
import torch
import tqdm

from doe.models.advanced import ChickenAdvanced

with open("/home/frank/Projects/DecoderTest/text.txt", "r", encoding="utf-8") as f:
    text: str = f.read()
chars = sorted(list(set(text)))
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string
vocabulary_size: int = len(chars)


DEVICE: str = "cuda"
SEED: int = 421

TRAIN_VAL_SPLIT: int = 0.8

batch_size: int = 8
epochs: int = 10000
step_size: int = 768  # Max num characters to look back.
embedding_dim: int = 512
n_blocks: int = 16
n_heads: int = 16
lr: float = 1e-4
dropout: float = 0.2
preview_size: int = 3000

# Defined functions to feed the data.
torch.manual_seed(SEED)
torch.set_default_device(DEVICE)
if DEVICE == "mps":
    MAX_MEMORY = 16  # In GB.
    torch.mps.set_per_process_memory_fraction(MAX_MEMORY / 64)

# Create a tensor to hold the encoded input.
data: torch.Tensor = torch.tensor(encode(text), dtype=torch.long, device=DEVICE)
print(data.shape, data.dtype)

# Now separate the data into Train and Valid.
split = TRAIN_VAL_SPLIT
train_data, valid_data = data[:int(len(data) * split)], data[int(len(data) * split):]
print(len(train_data), len(valid_data))


def get_data(mode: str = "train") -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate a small batch of data. Both input and target."""
    if mode == "train":
        data: torch.Tensor = train_data
    elif mode == "valid":
        data = valid_data
    else:
        raise NotImplementedError

    # print(len(data), 
        
    indices: torch.Tensor = torch.randint(len(data) - step_size, (batch_size, ))
    x: torch.Tensor = torch.stack([data[i: i + step_size] for i in indices])
    y: torch.Tensor = torch.stack([data[i + 1: i + step_size + 1] for i in indices])
    return x, y
    
    
x_batch, y_batch = get_data()
print(x_batch)
print(y_batch)



m = ChickenAdvanced(
    vocabulary_size=vocabulary_size,
    step_size=step_size,  # Max num characters to look back.
    embedding_dim=embedding_dim,
    n_blocks=n_blocks,
    n_heads=n_heads,
    dropout=dropout,
)


# Simple training loop.
optimizer = torch.optim.AdamW(m.parameters(), lr=lr)

for step_i in tqdm.tqdm(
    range(epochs),
    total=epochs
):
    # Get x and y.
    x_batch, y_batch = get_data("train")

    # Get loss.
    logits, loss = m(x_batch, y_batch)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
    if step_i % 1000 == 0:
        print(loss.item())
print(loss.item())

a = m.generate(torch.zeros((1, 1), dtype=torch.int), preview_size) 
print(decode(a[0].tolist()))