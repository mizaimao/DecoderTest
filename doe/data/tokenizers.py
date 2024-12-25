from typing import Callable, Dict, List, Tuple
from pathlib import Path
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
    

class MidiTokWrapper:

    def __init__(self, file_path: str):
        from miditok import REMI, TokenizerConfig
        from symusic import Score

        vol_size: int = 400


        # Creating a multitrack tokenizer, read the doc to explore all the parameters
        config = TokenizerConfig(num_velocities=16, use_chords=True, use_programs=True)
        self.tokenizer = REMI(config)
        self.midi = None

        # Loads a midi, converts to tokens, and back to a MIDI
        if Path(file_path).is_file():
            midi = Score(file_path)
            # calling the tokenizer will automatically detect MIDIs, paths and tokens
            self.tokens = self.tokenizer(midi)
            self.vocabulary_size: int = len(self.tokens)

        elif Path(file_path).is_dir():
            file_path = list(Path(file_path).glob("*.mid"))
            # midis = [self.tokenizer(x) for x in file_path]
            # self.tokens = []
            # for x in midis:
            #     self.tokens.extend(x)
            self.tokenizer.train(vocab_size=vol_size, files_paths=file_path)

            self.tokens = []
            for x in file_path:
                self.tokens.extend(
                    self.tokenizer.encode(Score(x))
                )
            self.vocabulary_size = len(self.tokens)
            # breakpoint()

        else:
            raise FileNotFoundError
        

    def encode(self, s: str) -> List[int]:
        if s is None:
            return self.tokens
        else:
            raise NotImplementedError
        
    def decode(self, numbers: List[int]):
        if len(numbers) == 1:
            numbers = numbers[0]
        converted_back_midi = self.tokenizer(
            numbers.cpu().numpy()
        )  # PyTorch, Tensorflow and Numpy tensors are supported # My ass

        return converted_back_midi




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
    
    elif name == "miditok":
        return MidiTokWrapper(
            file_path=file_path
        )

    else:
        raise NotImplementedError
