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
from torch import autograd


def train():
    train, val, test = get_dataloaders(period=365, batch_size=4, collate_fn=collate_fn)
    t_total = 600
    gpu = 5
    scheduler_args = {"num_warmup_steps": int(t_total * 0.1), "num_training_steps": t_total}
    for example in train:
        e = example
        break
    model = MMIL(scheduler_args=scheduler_args, use_mask=True)
    trainer = pl.Trainer(
        max_steps=t_total,
        gpus=[gpu],
        accumulate_grad_batches=1,
        track_grad_norm=2,
        gradient_clip_val=0.5,
        log_every_n_steps=10,
    )
    with autograd.detect_anomaly():
        trainer.fit(model, train, val)
    

if __name__ == "__main__":
   train() 