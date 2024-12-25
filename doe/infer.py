from pathlib import Path
import torch

from doe.models import get_models
from doe.configs.default import current_config, DefaultConfig, MIDIConfig
from doe.data.dataloaders import get_loader

DEVICE: str = "cuda"
config = current_config
model_loc: Path = Path(config.save_loc).joinpath("chicken.pt")
midi_save_loc: Path = Path(config.save_loc).joinpath("midichicken.mid")

torch.set_default_device(DEVICE)


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

if isinstance(config, MIDIConfig):

    tries: int = 0
    while True:
        try:
            converted_back_midi = loader.decode(generated)
            converted_back_midi.dump_midi(midi_save_loc)
            break
        except KeyError:
            tries += 1
            print(f"Key error on {tries}th trial, trying again...")
    print(f"Generated after {tries} tires.")

else:
    print(loader.decode(generated[0].tolist()))