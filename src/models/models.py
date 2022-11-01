import os.path

import torch
import numpy as np
import pytorch_lightning as pl
import wandb

from transformers import get_cosine_schedule_with_warmup
from PIL import Image

from definitions import ROOT_DIR
from src.configs.config_classes import TrainConfig
from src.models.encoders import UNet
from src.sampling.strategies import ddpm


class BaseDiffusion(pl.LightningModule):
    def __init__(self, cfg: TrainConfig):
        super().__init__()

        self.cfg = cfg
        self.encoder = UNet()

        self.lr = cfg.opt.lr
        self.w_decay = cfg.opt.w_decay
        self.scheduler = cfg.opt.scheduler
        self.warmup_steps = cfg.opt.n_warmup_steps

        alpha = 1 - torch.linspace(*self.cfg.opt.noise_range, self.cfg.opt.n_diffusion_steps)
        self.register_buffer("alpha_bar", torch.cumprod(alpha, dim=0))

        self.save_hyperparameters()

    @property
    def num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        if self.trainer.max_steps is not None and self.trainer.max_steps > 0:
            return self.trainer.max_steps

        limit_batches = self.trainer.limit_train_batches
        batches = len(self.trainer.datamodule.train_dataloader())
        batches = min(batches, limit_batches) if isinstance(limit_batches, int) else int(limit_batches * batches)

        num_devices = max(1, self.trainer.num_gpus, self.trainer.num_processes)
        if self.trainer.tpu_cores:
            num_devices = max(num_devices, self.trainer.tpu_cores)

        effective_accum = self.trainer.accumulate_grad_batches * num_devices
        total_steps = (batches // effective_accum) * self.trainer.max_epochs

        return total_steps

    def q_x0_xt(self, x0, t):
        mean = (self.alpha_bar[t] ** 0.5).reshape(-1, 1, 1, 1) * x0
        var = ((1 - self.alpha_bar[t]) ** 0.5).reshape(-1, 1, 1, 1)
        eps = torch.randn_like(x0)
        return mean + var * eps, eps

    def forward(self, batch):
        x0 = batch
        bs = len(x0)
        t = torch.randint(0, self.cfg.opt.n_diffusion_steps, (bs,), dtype=torch.long, device=x0.device)
        xt, noise = self.q_x0_xt(x0, t)
        noise_pred = self.encoder(xt, t)

        loss = torch.nn.functional.mse_loss(noise_pred, noise)

        return {
            "loss": loss,
        }

    def training_step(self, batch, batch_idx):
        model_out = self(batch)
        loss = model_out["loss"]

        # logging
        self.log("train_loss", loss.item(), on_step=True, prog_bar=True, logger=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        model_out = self(batch)
        loss = model_out["loss"]

        output = {"loss": loss.item()}

        return output

    def validation_epoch_end(self, validation_step_outputs):
        loss = np.mean([l["loss"] for l in validation_step_outputs])
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)

        if int(os.environ["LOCAL_RANK"]) == 0 and (self.current_epoch + 1) % self.cfg.opt.log_samples_every_epochs == 0:
            self._log_samples()

    def _log_samples(self):
        samples = ddpm(
            self.encoder,
            im_size=self.cfg.data.image_size,
            n_samples=4,
            n_diffusion_steps=self.cfg.opt.n_diffusion_steps,
            noise_range=self.cfg.opt.noise_range,
            use_gpu=self.cfg.opt.gpus != 0,
        )

        image = Image.new("RGB", size=(self.cfg.data.image_size * len(samples), self.cfg.data.image_size))
        for i, sample in enumerate(samples):
            image.paste(sample, (i * self.cfg.data.image_size, 0))

        image.save(os.path.join(ROOT_DIR, "data/results/samples.png"))
        self.logger.log_image(key="samples", images=[image])

    def configure_optimizers(self):
        print(f"Training steps: {self.num_training_steps}")
        optimizer = torch.optim.AdamW(self.encoder.parameters(), lr=self.lr, weight_decay=self.w_decay)
        if isinstance(self.warmup_steps, float):
            warmup_steps = self.num_training_steps * self.warmup_steps
        else:
            warmup_steps = self.warmup_steps

        if self.scheduler == "cosine":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=self.num_training_steps,
            )
            scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}

            return [optimizer], [scheduler]
        else:
            return optimizer
