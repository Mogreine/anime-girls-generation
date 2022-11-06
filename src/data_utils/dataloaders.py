import os
from typing import Optional, List

import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import transforms, datasets

from src.configs.config_classes import TrainConfig
from src.data_utils.utils import load_images


class AnimeFacesDataset(Dataset):
    def __init__(self, cfg: TrainConfig, image_files: List[str]):
        self.cfg = cfg
        self.image_files = image_files
        self.transforms = transforms.Compose([
            transforms.Resize((cfg.data.image_size, cfg.data.image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.cfg.data.image_folder, self.image_files[idx]))
        img = self.transforms(img)

        return img


class BaseDataModule(pl.LightningDataModule):
    def __init__(self, cfg: TrainConfig):
        super().__init__()
        self.batch_size = cfg.opt.batch_size
        self.num_workers = cfg.data.n_workers

        self.cfg = cfg

    def setup(self, stage: Optional[str] = None) -> None:
        img_files = os.listdir(self.cfg.data.image_folder)
        np.random.shuffle(img_files)

        self.train_ds = AnimeFacesDataset(self.cfg, img_files[:int(len(img_files) * 0.95)])
        self.val_ds = AnimeFacesDataset(self.cfg, img_files[int(len(img_files) * 0.95):])

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )
