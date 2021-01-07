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
from typing import Dict, Literal
import wandb
from sklearn.metrics import precision_recall_fscore_support
import random


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


def init_weights(*models):
    for model in models:
        for name, param in model.named_parameters():
            if "weight" in name or "bias" in name:
                param.data.uniform_(-0.1, 0.1)


class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ["dot", "general", "concat"]:
            raise ValueError(self.method, "is not an appropriate attention method.")
        self.hidden_size = hidden_size
        if self.method == "general":
            self.attn = nn.Linear(self.hidden_size, hidden_size)
        elif self.method == "concat":
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        energy = self.attn(
            torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)
        ).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        # Calculate the attention weights (energies) based on the given method
        if self.method == "general":
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == "concat":
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == "dot":
            attn_energies = self.dot_score(hidden, encoder_outputs)

        # Transpose max_length and batch_size dimensions
        attn_energies = attn_energies.t()

        # Return the softmax normalized probability scores (with added dimension)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)


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


class EncoderLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_layers: int = 1,
        dropout: float = 0,
        ignore_pad: bool = True,
        batch_first: bool = False,
    ):
        super(EncoderLSTM, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.ignore_pad = ignore_pad
        self.lstm = nn.LSTM(
            self.input_size,
            self.hidden_size,
            self.n_layers,
            batch_first=batch_first,
            dropout=dropout,
        )

    def forward(self, x, input_lengths=None):
        # input_lengths = key_padd_mask.shape[1] - key_padd_mask.sum(dim=1)
        if self.ignore_pad:
            x = torch.nn.utils.rnn.pack_padded_sequence(
                x, input_lengths, enforce_sorted=False
            )

        outputs, (ho, co) = self.lstm(x)

        if self.ignore_pad:
            outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)

        return outputs, (ho, co)


class DecoderLSTM(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        output_size: int,
        n_layers: int = 1,
        dropout=0.1,
        batch_first: bool = False,
    ):
        super(DecoderLSTM, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lstm = nn.LSTM(
            self.hidden_size,
            self.output_size,
            self.n_layers,
            batch_first=batch_first,
            dropout=dropout,
        )

    def forward(self, x, last_hidden, encoder_outputs = None):
        return self.lstm(x, last_hidden), None


class AttnDecoderLSTM(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        output_size: int,
        n_layers: int = 1,
        dropout: float = 0.1,
        batch_first: bool = False,
    ):
        super(AttnDecoderLSTM, self).__init__()

        # From https://pytorch.org/tutorials/beginner/deploy_seq2seq_hybrid_frontend_tutorial.html
        self.attn_model = "dot"
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        self.lstm = nn.LSTM(
            self.hidden_size,
            self.output_size,
            self.n_layers,
            batch_first=batch_first,
            dropout=dropout,
        )
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.attn = Attn(self.attn_model, hidden_size)

    def forward(self, x, last_hidden, encoder_outputs):
        # Note: we run this one step (word) at a time
        # Get embedding of current input word
        # Forward through unidirectional GRU
        rnn_output, hidden = self.lstm(x, last_hidden)
        # Calculate attention weights from the current GRU output
        attn_weights = self.attn(rnn_output, encoder_outputs)
        # Multiply attention weights to encoder outputs to get new "weighted sum" context vector
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        # Concatenate weighted context vector and GRU output using Luong eq. 5
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        # Predict next word using Luong eq. 6
        # output = self.out(concat_output).unsqueeze(0)
        # Lembrar que se der problema voltar com a linha de baixo e tirar o unsqueeze(0) da linha acima
        output = self.out(concat_output).unsqueeze(0)
        # output = F.softmax(output, dim=1).unsqueeze(0)
        # Return output and final hidden state
        return output, hidden, attn_weights



class Seq2SeqLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        enc_n_layers: int = 1,
        dec_n_layers: int = 1,
        dropout: float = 0.1,
        ignore_pad: bool = True,
        batch_first: bool = False,
        use_attention: bool = False,
        teacher_force: float = 0.0,
    ):
        super(Seq2SeqLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.enc_n_layers = enc_n_layers
        self.dec_n_layers = dec_n_layers
        self.dropout = dropout
        self.ignore_pad = ignore_pad
        self.batch_first = batch_first
        self.teacher_force = teacher_force
        self.encoder = EncoderLSTM(
            self.input_size,
            self.hidden_size,
            self.enc_n_layers,
            self.dropout,
            self.ignore_pad,
            self.batch_first,
        )
        Decoder = AttnDecoderLSTM if use_attention else DecoderLSTM
        print(f"Using {Decoder} decoder.")
        self.decoder = Decoder(
            self.hidden_size,
            self.output_size,
            self.dec_n_layers,
            self.dropout,
            self.batch_first,
        )

    # def forward(self, src, tgt, input_lengths):
    #     max_seq_size = src.size(0)
    #     # decoder_hidden = (ho, co)
    #     encoder_outputs, decoder_hidden = self.encoder(src, input_lengths)
    #     # co = torch.zeros(ho.shape, device=src.device, dtype=src.dtype)
    #     curr_input = tgt[0, :, :].unsqueeze(0)
    #     attn_weights = []
    #     for i in range(1, max_seq_size):
    #         decoder_output, decoder_hidden, attn_w = self.decoder(
    #             curr_input, decoder_hidden, encoder_outputs
    #         )
    #         # attn_weights.append(attn_w.detach().squeeze())
    #         teacher_force = random.random() < self.teacher_force
    #         curr_input = tgt[i, :, :].unsqueeze(0) if teacher_force else decoder_output
    #         # curr_input = tgt[i, :, :].unsqueeze(0)

    #     # attn_weights.append(attn_w.detach().squeeze())

    #     return decoder_hidden[0][-1].squeeze(), None
    #     # return decoder_output[-1], torch.stack(attn_weights, dim=-1).transpose(1, 2)

    def forward(self, src, tgt, input_lengths):
        max_seq_size = src.size(0)
        encoder_outputs, decoder_hidden = self.encoder(src, input_lengths)
        curr_input = tgt[0, :, :].unsqueeze(0)
        attn_weights = []
        for i in range(1, max_seq_size):
            decoder_output, decoder_hidden, attn_w = self.decoder(
                curr_input, decoder_hidden, encoder_outputs
            )
            # attn_weights.append(attn_w.detach().squeeze())
            teacher_force = random.random() < self.teacher_force
            curr_input = tgt[i, :, :].unsqueeze(0) if teacher_force else decoder_output
            # curr_input = tgt[i, :, :].unsqueeze(0)

        # attn_weights.append(attn_w.detach().squeeze())

        return decoder_hidden[0][-1].squeeze(), None
        # return decoder_output[-1], torch.stack(attn_weights, dim=-1).transpose(1, 2)

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
        vis_model: str = settings.VISUAL_MODEL,
        language_model: str = settings.LANGUAGE_MODEL,
        rnn_type: Literal["transformer", "lstm"] = "transformer",
        attention: bool = False,
        weight: bool = False,
        teacher_force: float = 0.0,
    ):
        super().__init__()
        self.scheduler_args = scheduler_args
        self.optimizer_args = optimizer_args
        self.ignore_pad = ignore_pad
        self.d_model = d_model
        self.weight = weight
        if rnn_type == "transformer":

            self.rnn = nn.Transformer(
                d_model=d_model,
                nhead=nhead,
                num_encoder_layers=num_encoder_layers,
                num_decoder_layers=num_decoder_layers,
            )
        elif rnn_type == "lstm":

            self.rnn = Seq2SeqLSTM(
                d_model,
                d_model,
                d_model,
                ignore_pad=self.ignore_pad,
                enc_n_layers=num_encoder_layers,
                dec_n_layers=num_decoder_layers,
                use_attention=attention,
                teacher_force=teacher_force,
            )

        else:

            raise NotImplementedError(f"The rnn type {self.rnn_type} is not implemented")

        self.rnn_type = rnn_type
        self.text_encoder = AutoModel.from_pretrained(language_model)
        self.visual_encoder = getattr(models, vis_model)(pretrained=True)
        self.visual_encoder.fc = nn.Identity()
        self.dropout = nn.Dropout(0.2)
        self.pos_encoder = PositionalEncoding(d_model)
        self.text_proj = nn.Linear(768, d_model)
        self.vis_proj = nn.Linear(512, d_model)
        self.classifier = nn.Linear(d_model, 2)
        self.vis_norm = nn.LayerNorm([settings.MAX_SEQ_LENGTH, 512])
        self.txt_norm = nn.LayerNorm([settings.MAX_SEQ_LENGTH, 768])
        self.relu = nn.ReLU()
        self.init_layers()
        self.mask = None
        self.use_mask = use_mask
        self.freeze_layers(
            self.text_encoder,
            self.visual_encoder,
        )

    def generate_square_subsequent_mask(self, sz, device):
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = mask.masked_fill(mask == 0, float("-inf")).masked_fill(
            mask == 1, float(0.0)
        )
        return mask

    def forward(self, batch):
        src, tgt, key_padd_mask, labels = batch
        textual_ftrs = [
            self.text_encoder(input_ids, attn_mask)[-1]
            for input_ids, attn_mask in zip(src[0], src[1])
        ]
        textual_ftrs = torch.stack(textual_ftrs) * math.sqrt(self.d_model)
        textual_ftrs = self.dropout(self.text_proj(textual_ftrs).transpose(0, 1))

        visual_ftrs = [self.visual_encoder(user_imgs_seq) for user_imgs_seq in tgt]
        visual_ftrs = torch.stack(visual_ftrs) * math.sqrt(
            self.d_model
        )  # [BATCH, SEQ, EMB]
        visual_ftrs = self.dropout(self.vis_proj(visual_ftrs).transpose(0, 1))  # [SEQ, BATCH, EMB]

        if not self.ignore_pad: 
            key_padd_mask = None

        if self.mask is None and self.use_mask:
            self.mask = self.generate_square_subsequent_mask(
                visual_ftrs.size(0), visual_ftrs.device
            )

        if self.rnn_type == "transformer":

            hidden = self.rnn(
                self.pos_encoder(textual_ftrs),
                self.pos_encoder(visual_ftrs),
                src_mask=self.mask,
                tgt_mask=self.mask,
                src_key_padding_mask=key_padd_mask,
                tgt_key_padding_mask=key_padd_mask,
                memory_key_padding_mask=key_padd_mask,
            ).transpose(0, 1)

            pooled_out = avg_pool(hidden, key_padd_mask)
            attn_weights = None

        elif self.rnn_type == "lstm":

            input_lengths = (
                key_padd_mask.shape[1] - key_padd_mask.sum(dim=1)
                if self.ignore_pad
                else None
            )
            pooled_out, attn_weights = self.rnn(textual_ftrs, visual_ftrs, input_lengths)

        else:
            raise NotImplementedError(f"The rnn type {self.rnn_type} is not implemented")

        logits = self.classifier(pooled_out)

        return logits, attn_weights

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), **self.optimizer_args)
        scheduler = get_linear_schedule_with_warmup(optimizer, **self.scheduler_args)
        return [optimizer], [scheduler]

    def training_step(self, train_batch, batch_idx):
        *_, labels = train_batch
        logits, _ = self(train_batch)
        if labels is not None:
            if self.weight:
                w = torch.tensor([0.6279, 0.3721], dtype=logits.dtype, device=logits.device)
                loss_fct = nn.CrossEntropyLoss(weight=w)
            else:
                loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 2), labels.view(-1))

        with torch.no_grad():
            preds = F.softmax(logits, dim=-1).argmax(dim=-1)
            acc = ((preds == labels).sum().float()) / len(labels)
            print()
            print(f"===>TRAIN PREDS: {preds}")
            print(f"===>TRAIN LABEL: {labels}")
            print(f"===>TRAIN ACCUR: {acc}")
            print()

        # self.logger.experiment.log({"train_loss": loss})
        # self.log("train_loss", loss)
        # self.log_dict({"train_loss": loss, "train_acc": acc})
        preds = preds.unsqueeze(0) if preds.dim() == 0 else preds
        return {"loss": loss, "preds": preds, "targets": labels}

    def training_epoch_end(self, outs):
        preds = []
        targets = []
        losses = []
        for out in outs:
            losses.append(out["loss"] * len(out["targets"]))
            targets.append(out["targets"])
            preds.append(out["preds"])

        preds = torch.cat(preds)
        targets = torch.cat(targets)
        loss = sum(losses) / len(targets)
        acc = ((preds == targets).sum().float()) / len(targets)
        print()
        print(f"===>TRAIN BATCH ACCUR: {acc}")
        print()
        self.log_dict({"train_loss": loss, "train_acc": acc})

    def validation_step(self, val_batch, batch_idx):
        *_, labels = val_batch
        logits, _ = self(val_batch)
        if labels is not None:
            if self.weight:
                w = torch.tensor([0.6279, 0.3721], dtype=logits.dtype, device=logits.device)
                loss_fct = nn.CrossEntropyLoss(weight=w)
            else:
                loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 2), labels.view(-1))

        # self.logger.experiment.log({"val_loss": loss})
        # self.log("val_loss", loss)
        preds = F.softmax(logits, dim=-1).argmax(dim=-1)
        acc = ((preds == labels).sum().float()) / len(labels)
        print()
        print(f"===>VAL PREDS: {preds}")
        print(f"===>VAL LABEL: {labels}")
        print(f"===>VAL ACCUR: {acc}")
        print()

        # self.logger.experiment.log({"train_loss": loss})
        # self.log("train_loss", loss)
        # self.log_dict({"val_loss": loss, "val_acc": acc})
        preds = preds.unsqueeze(0) if preds.dim() == 0 else preds
        return {"val_loss": loss, "preds": preds, "targets": labels}

    def validation_epoch_end(self, outs):
        preds = []
        targets = []
        losses = []
        for out in outs:
            losses.append(out["val_loss"] * len(out["targets"]))
            targets.append(out["targets"])
            preds.append(out["preds"])

        preds = torch.cat(preds)
        targets = torch.cat(targets)
        loss = sum(losses) / len(targets)
        acc = ((preds == targets).sum().float()) / len(targets)
        print()
        print(f"===>VAL BATCH ACCUR: {acc}")
        print()
        self.log_dict({"val_loss": loss, "val_acc": acc})

    def test_step(self, test_batch, batch_idx):
        *_, labels = test_batch
        logits, attn_weights = self(test_batch)
        preds = F.softmax(logits, dim=-1).argmax(dim=-1)
        probas = F.softmax(logits, dim=-1)

        if labels is not None:
            if self.weight:
                w = torch.tensor([0.6279, 0.3721], dtype=logits.dtype, device=logits.device)
                loss_fct = nn.CrossEntropyLoss(weight=w, reduction='none')
            else:
                loss_fct = nn.CrossEntropyLoss(reduction='none')
            loss = loss_fct(logits.view(-1, 2), labels.view(-1)).cpu()

        attn_weights = attn_weights.cpu().squeeze() if attn_weights is not None else None
        labels = labels.cpu().tolist()
        probas = probas.cpu().tolist()
        preds = preds.cpu().tolist()
        return (labels, probas, preds, attn_weights, loss)

    def test_epoch_end(self, outputs):
        labels = []
        probas = []
        preds = []
        losses = []
        source_indices = []
        target_indices = []
        for i in outputs:
            labels.extend(i[0])
            probas.extend(i[1])
            preds.extend(i[2])
        #     vals , source_seq_idx = i[3].topk(k=3, dim=2)
        #     target_seq_idx = vals.sum(dim=2).argmax(dim=1)
        #     target_indices.append(target_seq_idx)
        #     source_indices.append(source_seq_idx)
        #     losses.append(i[4])
        
        # source_indices = torch.cat(source_indices, dim=0)
        # target_indices = torch.cat(target_indices, dim=0)
        # losses = torch.cat(losses)
        # _, largest_losses = losses.topk(k=2)
        # _, lowest_losses = losses.topk(k=2, largest=False)

        self.logger.experiment.log(
            {
                "roc": wandb.plots.ROC(
                    np.array(labels), np.array(probas), ["Not Depressed", "Depressed"]
                )
            }
        )
        self.logger.experiment.log(
            {
                "cm": wandb.sklearn.plot_confusion_matrix(
                    np.array(labels), np.array(preds), ["Not Depressed", "Depressed"]
                )
            }
        )
        precision, recall, fscore, _ = precision_recall_fscore_support(
            labels, preds, average="binary"
        )
        self.log_dict({"precision": precision, "recall": recall, "fscore": fscore})

    def init_layers(self):
        init_weights(self.text_proj, self.vis_proj, self.classifier, self.vis_norm, self.txt_norm)
        # nn.init.constant_(self.classifier.weight.data, 0.42)

        # nn.init.uniform_(self.text_proj.weight.data, -0.1, 0.1)
        # nn.init.uniform_(self.text_proj.bias.data, -0.1, 0.1)

        # nn.init.normal_(self.text_proj.weight.data, 0, 0.02)
        # nn.init.zeros_(self.text_proj.bias.data)

        if self.rnn_type == "lstm":
            for name, param in self.rnn.named_parameters():
                if "weight" in name or "bias" in name:
                    param.data.uniform_(-0.1, 0.1)

        elif self.rnn_type == "transformer":
            for name, param in self.rnn.named_parameters():
                if "weight" in name:
                    if param.dim() > 1:
                        nn.init.xavier_uniform_(param.data)
                elif "bias" in name:
                    nn.init.constant_(param.data, 0.0)

    def freeze_layers(self, text_encoder: AutoModel, vis_encoder: ResNet):
        # visual, from https://discuss.pytorch.org/t/how-the-pytorch-freeze-network-in-some-layers-only-the-rest-of-the-training/7088/2
        for child in vis_encoder.children():
            for param in child.parameters():
                param.requires_grad = False

        # textual
        for name, param in text_encoder.named_parameters():
            param.requires_grad = False

        # Here we freeze all layers except the topmost layer.
        # for both textual and visual encoders