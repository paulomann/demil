import torch.nn as nn
import torch
from transformers import AutoModel, AutoTokenizer, PreTrainedModel
import pytorch_lightning as pl
from typing import Literal, List
from demil.utils import get_dataloaders
from transformers import PreTrainedTokenizer
from demil import settings
from torchvision import models
from torchvision.models.resnet import ResNet
from demil.data import collate_fn

train, val, test = get_dataloaders(period=365, batch_size=16, collate_fn=collate_fn)
for batch in train:
    example = batch
    break