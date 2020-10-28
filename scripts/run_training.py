import torch.nn as nn
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from typing import Literal, List
from demil.utils import get_dataloaders
from demil import settings
from demil.data import collate_fn
from demil.models import MMIL
import click


@click.command()
@click.option(
    "--gpu",
    required=True,
    help=f"ID of the GPU to run the training",
    type=click.INT
)
@click.option(
    "--name",
    required=True,
    help=f"Name of the training (for wandb)",
    type=click.STRING
)
@click.option(
    "--bsz",
    default=2,
    help=f"Batch Size",
    type=click.INT
)
@click.option(
    "--epochs",
    default=10,
    help=f"Number of epochs",
    type=click.INT
)
@click.option(
    "--use-mask",
    is_flag=True,
    help=f"Use mask to make element i not attend to every other element in the sequence",
)
def train(gpu: int, name: str, bsz: int, epochs: int, use_mask: bool):
    train, val, test = get_dataloaders(period=365, batch_size=bsz, collate_fn=collate_fn)
    wandb_logger = WandbLogger(project="demil", name=name)
    gradient_accumulation_steps = 1
    t_total = (len(train) // gradient_accumulation_steps) * epochs
    scheduler_args = {"num_warmup_steps": int(t_total * 0.1), "num_training_steps": t_total}
    model = MMIL(scheduler_args=scheduler_args, use_mask=use_mask)
    trainer = pl.Trainer(
        max_steps=t_total,
        logger=wandb_logger,
        gpus=[gpu],
        accumulate_grad_batches=gradient_accumulation_steps,
        track_grad_norm=2,
        gradient_clip_val=0.5,
        log_every_n_steps=2,
    )
    trainer.fit(model, train, val)

if __name__ == "__main__":
   train() 