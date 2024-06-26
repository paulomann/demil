import torch.nn as nn
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning import seed_everything
from typing import Literal, List
from demil.data import get_dataloaders
from demil import settings
from demil.data import collate_fn_mil
from demil.models import MMIL, MSIL
import click
from transformers import AutoTokenizer


@click.command()
@click.option(
    "--gpu", required=True, help=f"ID of the GPU to run the training", type=click.INT
)
@click.option("--name", help=f"Name of the training (for wandb)", type=click.STRING)
@click.option("--bsz", default=8, help=f"Batch Size", type=click.INT)
@click.option("--epochs", default=15, help=f"Number of epochs", type=click.INT)
@click.option(
    "--use-mask/--no-use-mask",
    is_flag=True,
    help=f"Use mask to make element i not attend to every other element in the sequence. It does not work well with the --ignore-pad flag (yields nan results)",
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
    "--ignore-pad/--no-ignore-pad",
    is_flag=True,
    help=f"Ignore padding elements in the sequence. Does not work well with the --use-mask flag (it yields nan values in the training)",
    default=True,
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
    "--correct-bias/--no-correct-bias",
    is_flag=True,
    help=f"Whether ot not to correct bias in Adam",
    default=True,
)
@click.option(
    "--period",
    default=-1,
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
    "--wandb/--no-wandb",
    is_flag=True,
    help=f"Whether to use wandb or not",
    default=True,
)
@click.option(
    "--rnn-type",
    default="transformer",
    help=f"Huggingface name of the model to be used",
    type=click.STRING,
)
@click.option(
    "--shuffle/--no-shuffle",
    is_flag=True,
    help=f"Whether to shuffle dataset's dataloader or not.",
    default=False,
)
@click.option(
    "--seed",
    default=42,
    help=f"Fix the seed of the random number generator.",
    type=click.INT,
)
@click.option(
    "--overfit",
    help=f"Overfit the dataset in a predetermined number of batches",
    type=click.FLOAT,
    default=0
)
@click.option(
    "--attention/--no-attention",
    is_flag=True,
    help=f"Whether to use attention in the LSTM decoder or not.",
    default=False,
)
@click.option(
    "--seq-len",
    help=f"The maximum sequence length",
    type=click.INT,
    default=settings.MAX_SEQ_LENGTH
)
@click.option(
    "--weight/--no-weight",
    is_flag=True,
    help=f"Whether to use weighted loss function or not.",
    default=False,
)
@click.option(
    "--teacher-force",
    help=f"Whether to use teacher forcing or not in the Seq2seq LSTM training.",
    default=0.0,
    type=click.FLOAT,
)
@click.option(
    "--augment-data/--no-augment-data",
    is_flag=True,
    help=f"Whether to augment data or not.",
    default=False,
)
@click.option(
    "--text/--no-text",
    is_flag=True,
    help=f"Whether to use the text modality or not.",
    default=True,
)
@click.option(
    "--visual/--no-visual",
    is_flag=True,
    help=f"Whether to use the visual modality or not.",
    default=True,
)
@click.option(
    "--pos-embedding",
    help=f"Whether to use 'absolute' positional encoding, or 'relative_key', or 'relative_key_query'.",
    default="absolute",
    type=click.STRING,
)
@click.option(
    "--timestamp/--no-timestamp",
    is_flag=True,
    help=f"Whether to use the timestamp as a relative Positional Encoding technique or not.",
    default=False,
)
@click.option(
    "--dataset",
    help=f"Which dataset to use (DeprUFF/DepressBR).",
    default="DeprUFF",
    type=click.STRING,
)
@click.option(
    "--shuffle-posts/--no-shuffle-posts",
    is_flag=True,
    help=f"Whether to randomize the order of posts or not.",
    default=False,
)
@click.option(
    "--mil/--no-mil",
    is_flag=True,
    help=f"Whether to randomize the order of posts or not.",
    default=True,
)
@click.option(
    "--multimodal-model",
    help=f"Name of the multimodal model. Empty by default. Valids: 'M-CLIP/XLM-Roberta-Large-Vit-B-16Plus'.",
    default="",
    type=click.STRING,
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
    wandb: bool,
    rnn_type: str,
    shuffle: bool,
    seed: int,
    overfit: float,
    attention: bool,
    seq_len: int,
    weight: bool,
    teacher_force: float,
    augment_data: bool,
    text: bool,
    visual: bool,
    pos_embedding: str,
    timestamp: bool,
    dataset: str,
    shuffle_posts: bool,
    mil: bool,
    multimodal_model: str
):
    seed_everything(seed)
    parameters = locals()
    settings.MAX_SEQ_LENGTH = seq_len
    print(f"====> Parameters: {parameters}")
    available_periods = [365, 212, 60, -1]
    if period not in available_periods:
        raise ValueError(
            f"Period of {period} is not valid. Please, use one of the following: {available_periods}"
        )
    if ignore_pad and use_mask:
        raise ValueError("You cannot use --ignore-pad and --use-mask at the same time.")

    settings.LANGUAGE_MODEL = multimodal_model if multimodal_model else language_model
    tokenizer = AutoTokenizer.from_pretrained(multimodal_model) if multimodal_model else AutoTokenizer.from_pretrained(language_model)
    train, val, test = get_dataloaders(
        period=period,
        batch_size=bsz,
        collate_fn=collate_fn_mil if mil else None,
        shuffle=shuffle,
        tokenizer=tokenizer,
        augment_data=augment_data,
        dataset=dataset,
        shuffle_posts=shuffle_posts,
        mil=mil,
        use_visual=visual,
        multimodal_model=multimodal_model
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
    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc",
        mode="max",
    )
    wandb_logger = None
    if wandb:
        wandb_logger = WandbLogger(project="demil", name=name, config=parameters)

    if mil:
        model = MMIL(
            scheduler_args=scheduler_args,
            optimizer_args=optimizer_args,
            use_mask=use_mask,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            d_model=d_model,
            ignore_pad=ignore_pad,
            language_model=language_model,
            vis_model=vis_model,
            rnn_type=rnn_type,
            attention=attention,
            weight=weight,
            teacher_force=teacher_force,
            text=text,
            visual=visual,
            pos_embedding=pos_embedding,
            timestamp=timestamp,
            multimodal_model=multimodal_model
        )
    else:
        model = MSIL(
            scheduler_args=scheduler_args,
            optimizer_args=optimizer_args,
            language_model=language_model,
            vis_model=vis_model,
            text=text,
            visual=visual,
            d_model=d_model,
            weight=weight,
            multimodal_model=multimodal_model
        )
    trainer = pl.Trainer(
        deterministic=True,
        max_steps=t_total,
        logger=wandb_logger if wandb else None,
        gpus=[gpu],
        accumulate_grad_batches=gradient_accumulation_steps,
        track_grad_norm=2,
        gradient_clip_val=gradient_clip_val,
        log_every_n_steps=log_every_n_steps,
        checkpoint_callback=checkpoint_callback,
        default_root_dir=settings.MODELS_PATH,
        overfit_batches=overfit
    )
    # print(pl.utilities.model_summary.ModelSummary(model, max_depth=-1))
    trainer.fit(model, train, val)
    trainer.test(dataloaders=test, ckpt_path='best')


if __name__ == "__main__":
    train()