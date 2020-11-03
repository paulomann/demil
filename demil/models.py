import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from transformers import AutoModel, AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from demil import settings
from torchvision import models
from torchvision.models.resnet import ResNet
import math
from typing import Dict


def avg_pool(data, input_lens: torch.BoolTensor = None):
    if input_lens is not None:
        return torch.stack(
            [
                torch.sum(data[i, l.sum() :, :], dim=0) / (l.size(0) - l.sum())
                for i, l in enumerate(input_lens)
            ]
        )
    else:
        return torch.sum(data, dim=1) / float(data.shape[1])


class PositionalEncoding(nn.Module):
    def __init__(
        self, d_model: int, dropout: float = 0.1, max_len: int = settings.MAX_SEQ_LENGTH
    ):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # print(x.size(), self.pe.size())
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class MMIL(pl.LightningModule):
    def __init__(
        self,
        d_model: int = 126,
        nhead: int = 2,
        num_encoder_layers: int = 1,
        num_decoder_layers: int = 1,
        ignore_pad: bool = False,
        scheduler_args: Dict[str, int] = None,
        optimizer_args: Dict[str, int] = None,
        use_mask: bool = False,
    ):
        super().__init__()
        self.scheduler_args = scheduler_args
        self.optimizer_args = optimizer_args
        self.ignore_pad = ignore_pad
        self.d_model = d_model
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
        )
        self.text_encoder = AutoModel.from_pretrained(settings.LANGUAGE_MODEL)
        self.visual_encoder = models.resnet34(pretrained=True)
        self.visual_encoder.fc = nn.Identity()
        self.pos_encoder = PositionalEncoding(d_model)
        self.text_proj = nn.Linear(768, d_model)
        self.vis_proj = nn.Linear(512, d_model)
        self.classifier = nn.Linear(d_model, 2)
        self.init_layers()
        self.vis_norm = nn.LayerNorm([settings.MAX_SEQ_LENGTH, 512])
        self.txt_norm = nn.LayerNorm([settings.MAX_SEQ_LENGTH, 768])
        self.relu = nn.ReLU()
        self.mask = None
        self.use_mask = use_mask
        self.partially_freeze_layers(self.text_encoder, self.visual_encoder)

    def generate_square_subsequent_mask(self, sz, device):
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def forward(self, batch):
        src, tgt, key_padd_mask, labels = batch
        textual_ftrs = [
            self.text_encoder(input_ids, attn_mask)[-1]
            for input_ids, attn_mask in zip(src[0], src[1])
        ]
        textual_ftrs = self.txt_norm(torch.stack(textual_ftrs) * math.sqrt(self.d_model))
        textual_ftrs = self.text_proj(textual_ftrs).transpose(0, 1)

        visual_ftrs = [self.visual_encoder(user_imgs_seq) for user_imgs_seq in tgt]
        visual_ftrs = self.vis_norm(
            torch.stack(visual_ftrs) * math.sqrt(self.d_model)
        )  # [BATCH, SEQ, EMB]
        visual_ftrs = self.vis_proj(visual_ftrs).transpose(0, 1)  # [SEQ, BATCH, EMB]
        src = self.pos_encoder(textual_ftrs)
        tgt = self.pos_encoder(visual_ftrs)
        if self.ignore_pad:
            key_padd_mask = None

        if self.mask is None and self.use_mask:
            self.mask = self.generate_square_subsequent_mask(
                visual_ftrs.size(0), visual_ftrs.device
            )

        hidden = self.transformer(
            src,
            tgt,
            src_mask=self.mask,
            tgt_mask=self.mask,
            src_key_padding_mask=key_padd_mask,
            tgt_key_padding_mask=key_padd_mask,
            memory_key_padding_mask=key_padd_mask,
        ).transpose(0, 1)

        hidden = self.relu(hidden)

        if self.ignore_pad:
            pooled_out = avg_pool(hidden)
        else:
            pooled_out = avg_pool(hidden, key_padd_mask)

        logits = self.classifier(pooled_out)

        return logits

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), **self.optimizer_args)
        scheduler = get_linear_schedule_with_warmup(optimizer, **self.scheduler_args)
        return [optimizer], [scheduler]

    def training_step(self, train_batch, batch_idx):
        *_, labels = train_batch
        logits = self(train_batch)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 2), labels.view(-1))

        with torch.no_grad():
            preds = F.softmax(logits, dim=1).argmax(dim=1)
            acc = ((preds == labels).sum().float()) / len(labels)

        # self.logger.experiment.log({"train_loss": loss})
        # self.log("train_loss", loss)
        self.log_dict({"train_loss": loss, "train_acc": acc})
        return {"loss": loss, "train_acc": acc}

    def validation_step(self, val_batch, batch_idx):
        *_, labels = val_batch
        logits = self(val_batch)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 2), labels.view(-1))

        # self.logger.experiment.log({"val_loss": loss})
        # self.log("val_loss", loss)
        preds = F.softmax(logits, dim=1).argmax(dim=1)
        acc = ((preds == labels).sum().float()) / len(labels)

        # self.logger.experiment.log({"train_loss": loss})
        # self.log("train_loss", loss)
        self.log_dict({"val_loss": loss, "val_acc": acc})
        return {"loss": loss, "val_acc": acc}

    def test_step(self, test_batch, batch_idx):
        *_, labels = batch
        logits = self(test_batch)
        _, preds = torch.max(out, 1)
        probas = F.softmax(out)
        self.logger.experiment.log(
            {
                "roc": wandb.plots.ROC(
                    labels.numpy(), probas.numpy(), ["Not Depressed", "Depressed"]
                )
            }
        )
        self.logger.experiment.sklearn.plot_confusion_matrix(
            labels.numpy(), preds.numpy(), ["Not Depressed", "Depressed"]
        )

    def init_layers(self):
        nn.init.normal_(self.text_proj.weight.data, 0, 0.02)
        nn.init.normal_(self.vis_proj.weight.data, 0, 0.02)
        nn.init.normal_(self.classifier.weight.data, 0, 0.02)
        nn.init.zeros_(self.text_proj.bias.data)
        nn.init.zeros_(self.vis_proj.bias.data)
        nn.init.zeros_(self.classifier.bias.data)

    def partially_freeze_layers(self, text_encoder, vis_encoder):
        # visual, from https://discuss.pytorch.org/t/how-the-pytorch-freeze-network-in-some-layers-only-the-rest-of-the-training/7088/2
        ct = 0
        for child in vis_encoder.children():
            ct += 1
            if ct < 7:
                for param in child.parameters():
                    param.requires_grad = False

        # textual
        for name, param in text_encoder.named_parameters():
            if "pooler" not in name and "11" not in name:
                param.requires_grad = False

        # Here we freeze all layers except the topmost layer.
        # for both textual and visual encoders