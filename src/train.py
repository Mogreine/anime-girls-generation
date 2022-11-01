import os
import wandb
import pyrallis
import pytorch_lightning as pl

from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from definitions import CONFIGS_DIR
from src.data_utils.dataloaders import BaseDataModule
from src.models.models import BaseDiffusion
from src.configs.config_classes import TrainConfig


@pyrallis.wrap(config_path=os.path.join(CONFIGS_DIR, "training_cfg.yaml"))
def train(cfg: TrainConfig):
    pl.seed_everything(cfg.seed)

    if cfg.opt.checkpoint_path:
        model = BaseDiffusion.load_from_checkpoint("wandb/test-run/files/best.ckpt")
    else:
        model = BaseDiffusion(cfg)

    datamodule = BaseDataModule(cfg)

    logger = WandbLogger(project="anime-girls", offline=cfg.offline_logging)
    lr_logger = LearningRateMonitor()

    checkpoint_callback = ModelCheckpoint(
        filename="best",
        dirpath=str(logger.experiment.dir),
        save_top_k=1,
        save_last=True,
        verbose=True,
        monitor="val_loss",
        mode="min",
    )

    trainer = pl.Trainer(
        strategy="ddp",
        precision=32,
        val_check_interval=cfg.opt.val_interval,
        accumulate_grad_batches=cfg.opt.accumulate_grad_batches,
        gpus=cfg.opt.gpus,
        log_every_n_steps=5,
        logger=logger,
        callbacks=[lr_logger, checkpoint_callback],
        deterministic=True,
        max_epochs=cfg.opt.n_epochs,
    )

    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    train()
