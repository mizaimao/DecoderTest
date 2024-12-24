from pathlib import Path
import torch

from doe.models import get_models
from doe.configs.default import DefaultConfig
from doe.data.dataloaders import get_loader

DEVICE: str = "cuda"
config: DefaultConfig = DefaultConfig()
model_loc: Path = Path(config.save_loc).joinpath("chicken.pt")

loader = get_loader("text", config=config)
model = get_models(
    "advanced", config=config, vocabulary_size=loader.get_vocabulary_size()
)

model.load_state_dict(
    torch.load(
        model_loc, weights_only=True
    )
)
model.eval()

# Now take a preview of the generated contents.
generated: torch.Tensor = model.generate(
    torch.zeros((1, 1), dtype=torch.int), config.preview_size
)
print(loader.decode(generated[0].tolist()))