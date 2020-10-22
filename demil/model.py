import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoModel, AutoTokenizer
from demil import settings
from torchvision import models
from torchvision.models.resnet import ResNet
import math


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
    def __init__(self, d_model: int, dropout:float = 0.1, max_len: int = 5000):
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
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class MMIL(pl.LightningModule):
    def __init__(
        self,
        d_model: int = 126,
        nhead: int = 2,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        ignore_pad: bool = False,
    ):
        super().__init__()
        self.ignore_pad = ignore_pad
        self.d_model = d_model
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
        )
        self.text_encoder = AutoModel.from_pretrained(settings.DEFAULT_MODEL)
        self.visual_encoder = models.resnet34(pretrained=True)
        self.visual_encoder.fc = nn.Identity()
        self.pos_encoder = PositionalEncoding(d_model)
        self.text_proj = nn.Linear(768, d_model)
        self.vis_proj = nn.Linear(512, d_model)
        self.classifier = nn.Linear(d_model, 2)

    def forward(self, batch):
        src, tgt, key_padd_mask, labels = batch
        textual_ftrs = [
            self.text_encoder(input_ids, attn_mask)[-1]
            for input_ids, attn_mask in zip(src[0], src[1])
        ]
        textual_ftrs = torch.stack(textual_ftrs) * math.sqrt(self.d_model)
        textual_ftrs = self.text_proj(textual_ftrs).transpose(0, 1)

        visual_ftrs = [self.visual_encoder(user_imgs_seq) for user_imgs_seq in tgt]
        visual_ftrs = torch.stack(visual_ftrs) * math.sqrt(self.d_model)
        visual_ftrs = self.vis_proj(visual_ftrs).transpose(0, 1)
        src = self.pos_encoder(textual_ftrs)
        tgt = self.pos_encoder(visual_ftrs)
        hidden = self.transformer(
            src,
            tgt,
            src_key_padding_mask=key_padd_mask,
            tgt_key_padding_mask=key_padd_mask,
        ).transpose(0, 1)
        if self.ignore_pad:
            pooled_out = avg_pool(hidden)
        else:
            pooled_out = avg_pool(hidden, key_padd_mask)

        logits = self.classifier(pooled_out)

        return logits

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-4)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        *_, labels = train_batch
        logits = self(train_batch)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 2), labels.view(-1))
        self.logger.experiment.log({"train_loss": loss})
        return loss

    def validation_step(self, val_batch, batch_idx):
        *_, labels = val_batch
        logits = self(val_batch)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 2), labels.view(-1))

        self.logger.experiment.log({"val_loss": loss})

        return loss

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
