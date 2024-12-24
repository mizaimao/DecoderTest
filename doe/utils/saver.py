import torch

def save(
        mode: str = "intermediate", 
        model: torch.nn.Module = None,
        loc: str = ""):
    assert model is not None
    if not loc:
        print("Not saving becasue location was not given.")
        return 
    torch.save(model.state_dict(), loc)