import os.path
from dataclasses import dataclass
from typing import Union, List, Optional
from pyrallis import field

from definitions import ROOT_DIR


@dataclass
class DataConfig:
    image_folder: str = field(default="data/anime-faces")
    image_size: int = field(default=64)
    interpolation: str = field(default="bicubic")
    n_workers: int = field(default=1)

    def __post_init__(self):
        self.image_folder = os.path.join(ROOT_DIR, self.image_folder)
        assert self.interpolation in ["bicubic", "nearest", "bilinear"]


@dataclass
class OptConfig:
    # Number of GPUs
    gpus: Union[int, List[int]] = field(default=8)
    # Batch size for training and validation
    batch_size: int = field(default=40)
    # Number of epochs for training
    n_epochs: int = field(default=10)
    # Number of steps to accumulate gradient for
    accumulate_grad_batches: int = field(default=7)
    # Run validation every val_interval steps
    val_interval: float = field(default=150_000)
    # Whether to resume from checkpoint or not
    checkpoint_path: str = field(default=None)

    # Number of warmup steps for training. Maybe either int or float
    n_warmup_steps: Union[int, float] = field(default=10_000)
    # Learning rate for AdamW optimizer
    lr: float = field(default=2e-4)
    # Weight decay for AdamW optimizer
    w_decay: float = field(default=0)
    # Scheduler type. Supports "cosine" and "slanted"
    scheduler: str = field(default="cosine")

    # Diffusion specific parameters
    # Number of diffusion steps
    n_diffusion_steps: int = field(default=1000)
    # Noise schedule
    noise_schedule: str = field(default="linear")
    # Noise min and max
    noise_range: List[float] = field(default_factory=lambda: [0.1, 0.1])

    # Samples logging interval
    log_samples_every_epochs: int = field(default=10)

    def __post_init__(self):
        if self.checkpoint_path is not None:
            self.checkpoint_path = os.path.join(ROOT_DIR, self.checkpoint_path)

        assert self.scheduler in [
            "cosine",
            "constant",
        ], f"Unknown scheduler: {self.scheduler}. Must be 'cosine' or 'constant'."


@dataclass
class ModelConfiguration:
    pass


@dataclass
class TrainConfig:
    """Config for training."""

    data: DataConfig = field(default_factory=DataConfig)
    model_config: ModelConfiguration = field(default_factory=ModelConfiguration)
    opt: OptConfig = field(default_factory=OptConfig)
    seed: int = 57
    offline_logging: bool = field(default=True)

