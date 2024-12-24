from typing import List, Tuple
import torch
import tqdm

from doe.models.advanced import ChickenAdvanced
from doe.data.dataloaders import TextLoader


DEVICE: str = "cuda"
SEED: int = 421

TRAIN_VAL_SPLIT: int = 0.8

batch_size: int = 8
epochs: int = 100
step_size: int = 128  # Max num characters to look back.
embedding_dim: int = 128
n_blocks: int = 4
n_heads: int = 4
lr: float = 3e-4
dropout: float = 0.2
preview_size: int = 200

# Defined functions to feed the data.
torch.manual_seed(SEED)
torch.set_default_device(DEVICE)
if DEVICE == "mps":
    MAX_MEMORY = 16  # In GB.
    torch.mps.set_per_process_memory_fraction(MAX_MEMORY / 64)

split = TRAIN_VAL_SPLIT

loader = TextLoader(
    step_size=step_size, batch_size=batch_size,file_path="/home/frank/Projects/DecoderTest/text.txt",train_val_split=TRAIN_VAL_SPLIT
)
vocabulary_size: int = loader.get_vocabulary_size()
x_batch, y_batch = loader.get_data()

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
    x_batch, y_batch = loader.get_data("train")

    # Get loss.
    logits, loss = m(x_batch, y_batch)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
    if step_i % 1000 == 0:
        print(loss.item())
print(loss.item())

a = m.generate(torch.zeros((1, 1), dtype=torch.int), preview_size) 
print(loader.decode(a[0].tolist()))