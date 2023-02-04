import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from transformers import AutoModel, AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from torch.optim import SGD
from demil import settings
from torchvision import models
from torchvision.models.resnet import ResNet
import math
from typing import Dict, Literal
import wandb
from sklearn.metrics import precision_recall_fscore_support
import random
from demil.transformer import BertMIL
from demil.utils import get_bert_config
from einops import reduce, rearrange
from multilingual_clip.pt_multilingual_clip import MultilingualCLIP
import open_clip
import fasttext
import fasttext.util

class PoolerWrapper():
    def __init__(self, out_features) -> None:
        self.dense = DenseWrapper(out_features=out_features)

class DenseWrapper():
    
    def __init__(self, out_features) -> None:
        self.out_features = out_features

class FastTextWrapper():

    def __init__(self):
        # TODO : To ensure platform agnostic running we need to change this
        # fasttext.util.download_model('en', if_exists='ignore')
        self.ft = fasttext.load_model('/home/arthurbittencourt/depression-demil/demil/scripts/cc.en.300.bin')   
        self.pooler = PoolerWrapper(self.ft.get_dimension())

    def __call__(self, input_ids, attn_mask):
        # TODO: Return something here
        print("Input Ids: ", input_ids)
        print("Attn_Mask: ", attn_mask)

    def tokenizer(self,
            caption,
            add_special_tokens=True,
            max_length=512,
            padding="max_length",
            return_tensors="pt",
            return_attention_mask=True,
            truncation=True,
        ):

        tokens = fasttext.tokenize(caption)

        input_ids = [self.ft.get_word_id(word) for word in tokens]

        length = min(len(input_ids), max_length)
        if (length < max_length): padding_length = max_length - len(input_ids) 
        else: padding_length = 0

        input_ids.extend([0]*padding_length)       
        input_ids = torch.tensor(input_ids[:max_length])

        # type_ids seems unused in the current implementation
        type_ids = torch.zeros(len(input_ids))

        attn_mask = [1] * length
        attn_mask.extend([0]*padding_length)
        attn_mask = torch.tensor(attn_mask)

        return {'input_ids':input_ids, 'token_type_ids':type_ids ,'attention_mask':attn_mask}


class TextMCLIP(MultilingualCLIP):

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
    
    def _forward(self, input_ids, attn_mask):
        # embs = self.transformer(input_ids, attn_mask)[0]
        # embs = (embs * attn_mask.unsqueeze(2)).sum(dim=1) / attn_mask.sum(dim=1)[:, None]
        # Or I can simply do this: 
        embs = self.transformer(input_ids, attn_mask)[1] # --> pooler_output (clf token)
        return self.LinearTransformation(embs)

    def __call__(self, input_ids, attn_mask):
        return (self._forward(input_ids, attn_mask),)

class VisualMCLIP(nn.Module):

    def __init__(self, multimodal_model: str):
        super(VisualMCLIP, self).__init__()
        self.visual_encoder = open_clip.create_model(settings.HUGGINGFACE_CLIP_TO_VISUAL[multimodal_model])
    
    def __call__(self, imgs):
        return self.visual_encoder.encode_image(imgs)



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


class MeanMIL(nn.Module):
    def __init__(self, d_model):
        super(MeanMIL, self).__init__()
        self.d_model = d_model
        self.linear = nn.Sequential(
            nn.Linear(self.d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(self.d_model // 2, self.d_model),
        )

    def forward(self, x):
        x = reduce(x, "b t e -> b e", "mean")
        logits = self.linear(x)
        return logits


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
        attn_energies[attn_energies == 0.0] = float("-Inf")

        # Return the softmax normalized probability scores (with added dimension)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)


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

    def forward(self, x, last_hidden, encoder_outputs=None):
        outputs, hidden = self.lstm(x, last_hidden)
        return (outputs, hidden, None)


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


class LSTM(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        output_size: int,
        n_layers: int = 1,
        dropout: float = 0.1,
        ignore_pad: bool = True,
        batch_first: bool = False,
    ):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout = dropout
        self.ignore_pad = ignore_pad
        self.batch_first = batch_first
        self.n_layers = n_layers
        self.lstm = nn.LSTM(
            self.hidden_size,
            self.output_size,
            self.n_layers,
            batch_first=self.batch_first,
            dropout=self.dropout,
        )

    def forward(self, x, input_lengths):

        if self.ignore_pad:
            x = torch.nn.utils.rnn.pack_padded_sequence(
                x, input_lengths, enforce_sorted=False
            )

        outputs, (ho, co) = self.lstm(x)
        return ho[-1], None


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

    def forward(self, src, tgt, input_lengths):
        # max_seq_size = src.size(0)
        bsz = src.size(1)
        encoder_outputs, (hidden, cell) = self.encoder(src, input_lengths)
        curr_input = tgt[0, 0, :].unsqueeze(0)
        attn_weights = []
        out = []
        for j in range(0, bsz):

            max_seq_size = input_lengths[j]
            ho, co = hidden[:, j, :].unsqueeze(0), cell[:, j, :].unsqueeze(0)

            for i in range(1, max_seq_size):
                decoder_output, (ho, co), attn_w = self.decoder(
                    curr_input.unsqueeze(0),
                    (ho, co),
                    encoder_outputs[:, j, :].unsqueeze(1),
                )
                # attn_weights.append(attn_w.detach().squeeze())
                teacher_force = random.random() < self.teacher_force
                curr_input = (
                    tgt[i, j, :].unsqueeze(0)
                    if teacher_force
                    else decoder_output.squeeze(0)
                )
                # curr_input = tgt[i, :, :].unsqueeze(0)

            decoder_output, (ho, co), attn_w = self.decoder(
                curr_input.unsqueeze(0), (ho, co), encoder_outputs[:, j, :].unsqueeze(1)
            )

            out.append(decoder_output.squeeze(0))

        # attn_weights.append(attn_w.detach().squeeze())

        return torch.cat(out), None

# Multimodal Single instance learning
class MSIL(pl.LightningModule):
    def __init__(
        self,
        scheduler_args: Dict[str, int] = None,
        optimizer_args: Dict[str, int] = None,
        vis_model: str = settings.VISUAL_MODEL,
        language_model: str = settings.LANGUAGE_MODEL,
        text: bool = True,
        visual: bool = True,
        d_model: int = 512,
        weight: bool = False,
        multimodal_model: str = ""
    ):
        
        super().__init__()
        self.scheduler_args = scheduler_args
        self.optimizer_args = optimizer_args
        self.dropout = nn.Dropout(0.2)
        self.d_model = d_model
        self.classifier = nn.Linear(self.d_model, 2)
        self.relu = nn.ReLU()
        self.text = text
        self.visual = visual
        self.weight = weight
        self.multimodal_model = multimodal_model

        self.text_encoder, self.visual_encoder = self.init_encoders(
            self.text, self.visual, vis_model, language_model, self.multimodal_model
        )
        self.txt_norm, self.vis_norm = self.init_norm(
            self.text_encoder, self.visual_encoder, self.multimodal_model
        )
        self.text_proj, self.vis_proj = self.init_proj(
            self.text_encoder, self.visual_encoder, self.multimodal_model, d_model
        )

        self.init_layers()
        # self.partially_freeze_layers(
        #     self.text_encoder,
        #     self.visual_encoder,
        # )
        self.freeze_layers(
            self.text_encoder,
            self.visual_encoder,
        )
        self.save_hyperparameters()

    def init_encoders(self, text: bool, visual: bool, vis_model: str, language_model: str, multimodal_model: str):
        
        visual_encoder, text_encoder = None, None
        if multimodal_model:
            # visual_encoder = open_clip.create_model(settings.HUGGINGFACE_CLIP_TO_VISUAL[multimodal_model])
            visual_encoder = VisualMCLIP(multimodal_model)
            text_encoder = TextMCLIP.from_pretrained(multimodal_model)
        if language_model == 'fasttext':
            visual_encoder = getattr(models, vis_model)(pretrained=True)
            visual_encoder.fc = nn.Identity()
            text_encoder = FastTextWrapper()
        else:
            visual_encoder = getattr(models, vis_model)(pretrained=True)
            visual_encoder.fc = nn.Identity()
            text_encoder = AutoModel.from_pretrained(language_model)
        
        return (text_encoder, visual_encoder)

    def init_norm(self, text_encoder, visual_encoder, multimodal_model):
        if multimodal_model:
            txt_norm = nn.LayerNorm(text_encoder.LinearTransformation.out_features)
            vis_norm = nn.LayerNorm(self.text_encoder.LinearTransformation.out_features)
        else:
            txt_norm = nn.LayerNorm(text_encoder.pooler.dense.out_features)
            vis_norm = nn.LayerNorm(512)

        return (txt_norm, vis_norm)

    def init_proj(self, text_encoder, visual_encoder, multimodal_model, d_model):
        if multimodal_model:
            text_proj = nn.Linear(self.text_encoder.LinearTransformation.out_features, d_model)
            vis_proj = nn.Linear(self.text_encoder.LinearTransformation.out_features, d_model)
        else:
            text_proj = nn.Linear(self.text_encoder.pooler.dense.out_features, d_model)
            vis_proj = nn.Linear(512, d_model)

        return (text_proj, vis_proj)

    def forward(self, batch):
        input_ids, attn_mask, imgs, _, labels = batch
        attn_weights = None

        textual_ftrs = 1
        visual_ftrs = 1
        if self.text:

            textual_ftrs = self.text_encoder(input_ids.squeeze(1), attn_mask.squeeze(1))[-1]
            textual_ftrs = self.dropout(self.text_proj(self.txt_norm(textual_ftrs)))

        if self.visual:

            visual_ftrs = self.visual_encoder(imgs)
            visual_ftrs = self.dropout(self.vis_proj(self.vis_norm(visual_ftrs)))

        x = textual_ftrs * visual_ftrs

        logits = self.classifier(x)

        return logits, attn_weights

    def configure_optimizers(self):
        print(f"Optimizer args: {self.optimizer_args}")
        optimizer = AdamW(self.parameters(), **self.optimizer_args)

        if self.scheduler_args:
            scheduler = get_linear_schedule_with_warmup(optimizer, **self.scheduler_args)
            return [optimizer], [scheduler]
        
        return optimizer

    def training_step(self, train_batch, batch_idx):
        *_, labels = train_batch
        logits, _ = self(train_batch)
        if labels is not None:
            if self.weight:
                w = torch.tensor([1.47, 1], dtype=logits.dtype, device=logits.device)
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
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 2), labels.view(-1))

        preds = F.softmax(logits, dim=-1).argmax(dim=-1)
        acc = ((preds == labels).sum().float()) / len(labels)
        print()
        print(f"===>VAL PREDS: {preds}")
        print(f"===>VAL LABEL: {labels}")
        print(f"===>VAL ACCUR: {acc}")
        print()

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
        *_, usernames, labels = test_batch
        logits, attn_weights = self(test_batch)
        preds = F.softmax(logits, dim=-1).argmax(dim=-1)
        probas = F.softmax(logits, dim=-1)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(reduction="none")
            loss = loss_fct(logits.view(-1, 2), labels.view(-1)).cpu()

        labels = labels.cpu().tolist()
        probas = probas.cpu().tolist()
        preds = preds.cpu().tolist()
        loss = loss.cpu().tolist()
        # return (labels, probas, preds, attn_weights, loss, inpt)
        return (labels, probas, preds, loss, usernames)

    def test_epoch_end(self, outputs):
        labels = []
        probas = []
        preds = []
        loss = []
        usernames = []
        for i in outputs:
            labels.extend(i[0])
            probas.extend(i[1])
            preds.extend(i[2])
            loss.extend(i[3])
            usernames.extend(i[4])

        new_labels = []
        new_preds = []
        new_probas = []

        uniq_usernames = set(usernames)
        for name in uniq_usernames:
            indices = [i for i, x in enumerate(usernames) if x == name]
            new_labels.append(labels[indices[0]])
            user_probas = [probas[idx][0] for idx in indices]
            proba = sum(user_probas)/len(user_probas)
            new_preds.append(int(proba > 0.5))
            new_probas.append([1-proba, proba])

        preds = new_preds
        labels = new_labels
        probas = new_probas

        print()
        print(f"===>TEST PREDS: {preds}")
        print(f"===>TEST LABEL: {labels}")
        print()

        if self.logger:
            
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

        precision, recall, fscore, _ = precision_recall_fscore_support(labels, preds, average="binary")
        self.log_dict({"precision": precision, "recall": recall, "fscore": fscore, "test_loss": sum(loss)/len(loss)})
        print(f"Precision: {precision}\nRecall: {recall}\nFscore: {fscore}")

    def init_layers(self):
        init_weights(
            self.text_proj, self.vis_proj, self.classifier, self.vis_norm, self.txt_norm
        )

    def partially_freeze_layers(self, text_encoder: AutoModel, vis_encoder: ResNet):
        ct = 0
        for child in vis_encoder.children():
            ct += 1
            if ct < 7:
                for param in child.parameters():
                    param.requires_grad = False

        for child in text_encoder.embeddings.children():
            for param in child.parameters():
                param.requires_grad = False

        ct = 0
        for child in text_encoder.encoder.layer.children():
            ct +=1 
            if ct < 7:
                for param in child.parameters():
                    param.requires_grad = False

    def freeze_layers(self, text_encoder, vis_encoder):
        for child in vis_encoder.children():
            for param in child.parameters():
                param.requires_grad = False

        # textual
        if hasattr(text_encoder, "named_parameters"):
            for name, param in text_encoder.named_parameters():
                param.requires_grad = False

        if hasattr(text_encoder, "pooler") and hasattr(text_encoder.pooler, "named_parameters"):
            for name, param in text_encoder.pooler.named_parameters():
                param.requires_grad = True

        elif self.multimodal_model == "M-CLIP/XLM-Roberta-Large-Vit-B-16Plus":
            for name, param in text_encoder.LinearTransformation.named_parameters():
                param.requires_grad = True


class MMIL(MSIL):
    def __init__(
        self,
        nhead: int = 2,
        num_encoder_layers: int = 1,
        num_decoder_layers: int = 1,
        ignore_pad: bool = False,
        use_mask: bool = False,
        rnn_type: Literal["transformer", "lstm"] = "transformer",
        attention: bool = False,
        teacher_force: float = 0.0,
        pos_embedding: str = "absolute",
        timestamp: bool = False,
        *args,
        **kwargs,
    ):
        super(MMIL, self).__init__(*args, **kwargs)
        self.ignore_pad = ignore_pad
        self.rnn_type = rnn_type
        self.mask = None
        self.use_mask = use_mask
        self.use_timestamp = timestamp
        
        if rnn_type == "transformer":

            if not (text and visual):
                raise ValueError(
                    "With seq2seq models we need to set --text and --visual modalities together."
                )

            self.rnn = nn.Transformer(
                d_model=self.d_model,
                nhead=nhead,
                num_encoder_layers=num_encoder_layers,
                num_decoder_layers=num_decoder_layers,
            )
        elif rnn_type == "seq2seq":
            if not (text and visual):
                raise ValueError(
                    "With seq2seq models we need to set --text and --visual modalities together."
                )

            self.rnn = Seq2SeqLSTM(
                self.d_model,
                self.d_model,
                self.d_model,
                ignore_pad=self.ignore_pad,
                enc_n_layers=num_encoder_layers,
                dec_n_layers=num_decoder_layers,
                use_attention=attention,
                teacher_force=teacher_force,
            )
        elif rnn_type == "lstm":
            self.rnn = LSTM(self.d_model, self.d_model, n_layers=1, ignore_pad=self.ignore_pad)
        elif rnn_type == "bert":
            bert_config = get_bert_config()
            bert_config.hidden_size = self.d_model
            bert_config.num_attention_heads = nhead
            bert_config.num_hidden_layers = num_encoder_layers
            bert_config.position_embedding_type = pos_embedding
            self.rnn = BertMIL(bert_config)
        elif rnn_type == "mean":
            self.rnn = MeanMIL(self.d_model)
        else:

            raise NotImplementedError(f"The rnn type {self.rnn_type} is not implemented")

        self.init_rnn_layers()

    def generate_square_subsequent_mask(self, sz, device):
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = mask.masked_fill(mask == 0, float("-inf")).masked_fill(
            mask == 1, float(0.0)
        )
        return mask

    def forward(self, batch):
        src, tgt, key_padd_mask, timestamps, labels = batch

        textual_ftrs = 1
        visual_ftrs = 1
        if self.text:

            textual_ftrs = [
                self.text_encoder(input_ids, attn_mask)[-1]
                for input_ids, attn_mask in zip(src[0], src[1])
            ]
            textual_ftrs = torch.stack(textual_ftrs) * math.sqrt(self.d_model)
            textual_ftrs = self.dropout(
                self.text_proj(self.txt_norm(textual_ftrs).transpose(0, 1))
            )

        if self.visual:

            visual_ftrs = [self.visual_encoder(user_imgs_seq) for user_imgs_seq in tgt]
            visual_ftrs = torch.stack(visual_ftrs) * math.sqrt(
                self.d_model
            )  # [BATCH, SEQ, EMB]
            visual_ftrs = self.dropout(
                self.vis_proj(self.vis_norm(visual_ftrs).transpose(0, 1))
            )  # [SEQ, BATCH, EMB]

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

            x = textual_ftrs * visual_ftrs
            pooled_out, attn_weights = self.rnn(x, input_lengths.to("cpu"))

        elif self.rnn_type == "bert":

            x = (textual_ftrs * visual_ftrs).transpose(0, 1)
            mask = 1 - key_padd_mask.type(x.dtype)
            if self.use_timestamp:
                pooled_out, attn_weights = self.rnn(
                    x, attention_mask=mask, timestamps=timestamps
                )
            else:
                pooled_out, attn_weights = self.rnn(x, attention_mask=mask)
        elif self.rnn_type == "mean":
            x = (textual_ftrs * visual_ftrs).transpose(0, 1)
            attn_weights = None
            pooled_out = self.rnn(x)
        else:
            raise NotImplementedError(f"The rnn type {self.rnn_type} is not implemented")

        logits = self.classifier(pooled_out)

        return logits, attn_weights

    def test_step(self, test_batch, batch_idx):
        *_, labels = test_batch
        logits, attn_weights = self(test_batch)
        preds = F.softmax(logits, dim=-1).argmax(dim=-1)
        probas = F.softmax(logits, dim=-1)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(reduction="none")
            loss = loss_fct(logits.view(-1, 2), labels.view(-1)).cpu()

        # attn_weights = attn_weights.cpu().squeeze() if attn_weights is not None else None
        labels = labels.cpu().tolist()
        probas = probas.cpu().tolist()
        preds = preds.cpu().tolist()
        loss = loss.cpu().tolist()
        # return (labels, probas, preds, attn_weights, loss, inpt)
        return (labels, probas, preds, loss)

    def test_epoch_end(self, outputs):
        labels = []
        probas = []
        preds = []
        loss = []
        for i in outputs:
            labels.extend(i[0])
            probas.extend(i[1])
            preds.extend(i[2])
            loss.extend(i[3])

        print()
        print(f"===>TEST PREDS: {preds}")
        print(f"===>TEST LABEL: {labels}")
        print()

        precision, recall, fscore, _ = precision_recall_fscore_support(
            labels, preds, average="binary"
        )
        self.log_dict({"precision": precision, "recall": recall, "fscore": fscore, "test_loss": sum(loss)/len(loss)})
        print(f"Precision: {precision}\nRecall: {recall}\nFscore: {fscore}")

    def init_rnn_layers(self):
        if self.rnn_type == "transformer":
            for name, param in self.rnn.named_parameters():
                if "weight" in name:
                    if param.dim() > 1:
                        nn.init.xavier_uniform_(param.data)
                elif "bias" in name:
                    nn.init.constant_(param.data, 0.0)
        elif self.rnn_type == "lstm":
            init_weights(self.rnn)