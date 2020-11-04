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
    "--gpu", required=True, help=f"ID of the GPU to run the training", type=click.INT
)
@click.option("--name", help=f"Name of the training (for wandb)", type=click.STRING)
@click.option("--bsz", default=8, help=f"Batch Size", type=click.INT)
@click.option("--epochs", default=15, help=f"Number of epochs", type=click.INT)
@click.option(
    "--use-mask",
    is_flag=True,
    help=f"Use mask to make element i not attend to every other element in the sequence",
)
@click.option(
    "--gradient-clip-val", default=0.5, help=f"Norm to clip gradients", type=click.FLOAT
)
@click.option("--log-every-n-steps", default=2, help=f"Log every n steps", type=click.INT)
@click.option(
    "--nhead",
    default=2,
    help=f"Number of heads in the Transformer architecture",
    type=click.INT,
)
@click.option(
    "--num-encoder-layers", default=1, help=f"Number of encoder layers", type=click.INT
)
@click.option(
    "--num-decoder-layers", default=1, help=f"Number of decoder layers", type=click.INT
)
@click.option("--d-model", default=126, help=f"d_model", type=click.INT)
@click.option(
    "--ignore-pad",
    is_flag=True,
    help=f"Use mask to make element i not attend to every other element in the sequence",
)
@click.option("--lr", default=2e-4, help=f"Learning Rate", type=click.FLOAT)
@click.option("--b1", default=0.9, help=f"AdamW b1 beta parameter", type=click.FLOAT)
@click.option("--b2", default=0.999, help=f"AdamW b2 beta parameter", type=click.FLOAT)
@click.option(
    "--eps",
    default=1e-6,
    help=f"Adam's epsilon for numerical stability",
    type=click.FLOAT,
)
@click.option(
    "--weight-decay", default=0, help=f"Decoupled weight decay to apply", type=click.FLOAT
)
@click.option(
    "--correct-bias",
    is_flag=True,
    help=f"Whether ot not to correct bias in Adam",
    default=True,
)
@click.option(
    "--period",
    default=365,
    help=f"The observation period of the DepressionCorpus data.",
    type=click.INT,
)
@click.option(
    "--language-model",
    default=settings.LANGUAGE_MODEL,
    help=f"Huggingface name of the model to be used",
    type=click.STRING,
)
@click.option(
    "--vis-model",
    default=settings.VISUAL_MODEL,
    help=f"Name of the visual model as in torchvision.models. Ex: 'resnet34'",
    type=click.STRING,
)
@click.option(
    "--txt-freeze-n-layers",
    default=10,
    help=f"The number of layers to freeze in the textual encoder. Maximum is 11 (number of transformer layers)",
    type=click.INT,
)
@click.option(
    "--vis-freeze-n-layers",
    default=7,
    help=f"The number of layers to freeze in the visual encoder. Maximum is 10 (number of blocks in the ResNet network)",
    type=click.INT,
)
def train(
    gpu: int,
    name: str,
    bsz: int,
    epochs: int,
    use_mask: bool,
    gradient_clip_val: float,
    log_every_n_steps: int,
    nhead: int,
    num_encoder_layers: int,
    num_decoder_layers: int,
    d_model: int,
    ignore_pad: bool,
    lr: float,
    b1: float,
    b2: float,
    eps: float,
    weight_decay: float,
    correct_bias: bool,
    period: int,
    language_model: str,
    vis_model: str,
    txt_freeze_n_layers: int,
    vis_freeze_n_layers: int,
):
    parameters = locals()
    available_periods = [365, 212, 60]
    if period not in available_periods:
        raise ValueError(
            f"Period of {period} is not valid. Please, use one of the following: {available_periods}"
        )
    settings.LANGUAGE_MODEL = language_model
    tokenizer = AutoTokenizer.from_pretrained(language_model)
    train, val, test = get_dataloaders(
        period=period, batch_size=bsz, collate_fn=collate_fn, tokenizer=tokenizer
    )
    gradient_accumulation_steps = 1
    t_total = (len(train) // gradient_accumulation_steps) * epochs
    scheduler_args = {
        "num_warmup_steps": int(t_total * 0.1),
        "num_training_steps": t_total,
    }
    optimizer_args = {
        "lr": lr,
        "betas": (b1, b2),
        "eps": eps,
        "weight_decay": weight_decay,
        "correct_bias": correct_bias,
    }
    parameters.update(scheduler_args)
    wandb_logger = WandbLogger(project="demil", name=name, config=parameters)
    model = MMIL(
        scheduler_args=scheduler_args,
        optimizer_args=optimizer_args,
        use_mask=use_mask,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        d_model=d_model,
        ignore_pad=ignore_pad,
        vis_freeze_n_layers=vis_freeze_n_layers,
        txt_freeze_n_layers=txt_freeze_n_layers,
        language_model=language_model,
        vis_model=vis_model
    )
    trainer = pl.Trainer(
        max_steps=t_total,
        logger=wandb_logger,
        gpus=[gpu],
        accumulate_grad_batches=gradient_accumulation_steps,
        track_grad_norm=2,
        gradient_clip_val=gradient_clip_val,
        log_every_n_steps=log_every_n_steps,
    )
    trainer.fit(model, train, val)


if __name__ == "__main__":
    train()
