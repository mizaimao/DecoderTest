from typing import List, Tuple
from pathlib import Path
import tqdm

import torch

from doe.models import get_models
from doe.models.blocks import get_valid_loss
from doe.data.dataloaders import get_loader
from doe.configs.default import DefaultConfig
from doe.utils.saver import save


DEVICE: str = "cuda"
config: DefaultConfig = DefaultConfig()


# Defined functions to feed the data.
torch.manual_seed(config.seed)
torch.set_default_device(DEVICE)
if DEVICE == "mps":
    MAX_MEMORY = 16  # In GB.
    torch.mps.set_per_process_memory_fraction(MAX_MEMORY / 64)


loader = get_loader("text", config=config)
model = get_models(
    "advanced", config=config, vocabulary_size=loader.get_vocabulary_size()
)
epochs: int = config.epochs
print_every: int = config.print_every
preview_size: int = config.preview_size
save_loc: Path = Path(config.save_loc)
last_loss: float = float("inf")


# Simple training loop.
optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

for step_i in tqdm.tqdm(range(epochs), total=epochs):
    # Start iteration with data.
    x_batch, y_batch = loader.get_data("train")

    # Calculate loss and back prop.
    logits, loss = model(x_batch, y_batch)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    # Validation and saving.
    if step_i % print_every == 0 or step_i == epochs - 1:
        val_loss: float = get_valid_loss(model=model, loader=loader)

        print("Train {:.04f}, Validation {:.04f}".format(loss.item(), val_loss.item()))
        if val_loss < last_loss:
            last_loss = val_loss
            save(
                model=model,
                loc=save_loc.joinpath(
                    "final.pt" if step_i == epochs - 1 else "chicken.pt"
                ),
            )


# Now take a preview of the generated contents.
generated: torch.Tensor = model.generate(
    torch.zeros((1, 1), dtype=torch.int), preview_size
)
print(loader.decode(generated[0].tolist()))