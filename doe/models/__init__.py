from doe.models.advanced import ChickenAdvanced
from doe.configs.default import DefaultConfig


def get_models(
    name: str = "advanced", config: DefaultConfig = None, vocabulary_size: int = None
):
    assert vocabulary_size is not None
    model = None
    if name == "advanced":
        model = ChickenAdvanced(
            vocabulary_size=vocabulary_size,
            step_size=config.step_size,  # Max num characters to look back.
            embedding_dim=config.embedding_dim,
            n_blocks=config.n_blocks,
            n_heads=config.n_heads,
            dropout=config.dropout,
        )
    else:
        raise NotImplementedError

    return model
