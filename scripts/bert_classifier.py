'''
    Code modified from https://github.com/adap/flower/blob/d1eb90f74714a9c10ddbeefb767b56be7b61303d/examples/quickstart_huggingface/client.py#L11
'''
import argparse

import pandas as pd
import os.path

from collections import OrderedDict
import warnings

import flwr as fl
import torch
import numpy as np

from torch.utils.data import DataLoader

from datasets import Dataset

from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification
from transformers import AdamW, Adafactor
from evaluate import load

import bert_utils

warnings.filterwarnings("ignore", category=UserWarning)

def get_args():
    parser = argparse.ArgumentParser(description="Creating a bert client for federated learning")
    parser.add_argument("--gpu", type=int, default=7, help="wich device this client will use")
    parser.add_argument("--checkpoint", type=str, default="bert-base-uncased", help="what checkpoint to use, leave blank for bert-base-uncased")
    parser.add_argument("--epochs", type=int, default=1, help="how many epochs to run on this client")
    parser.add_argument("--experiment_name", type=str, required=True)

    args = parser.parse_args()
    return args

args = get_args()

DEVICE = torch.device(f"cuda:{args.gpu}")
CHECKPOINT = args.checkpoint
EPOCHS = args.epochs
LOGGER = bert_utils.ExperimentLogger(args.experiment_name) 

def train(net, trainloader, valloader, epochs, lr, logger=LOGGER):
    optimizer = AdamW(net.parameters(), lr=lr)  # Adafactor(net.parameters(), warmup_init=True) 
    net.train()
    for i in range(epochs):
        for batch in trainloader:
            if 'username' in batch: batch.pop('username')
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = net(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        metrics = bert_utils.test(net, valloader, DEVICE, logger=LOGGER, source=f"Epoch {i+1}")
        metrics['train_loss'] = loss.item()   

        print(f"Epoch {i+1}: {metrics}")

def main():

    net = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT, num_labels=2).to(DEVICE)

    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
    trainloader, testloader, valloader = bert_utils.load_data(dataset='eRisk2021', partition=False, tokenizer=tokenizer)

    # Log baseline
    baseline_metrics = bert_utils.test(net, testloader, DEVICE, logger=LOGGER, source="Baseline")
    print(f"Baseline: {baseline_metrics}")

    # Start Training, three times incrementing the learning rate
    print("Sarting Training...")
    train(net, trainloader, valloader, EPOCHS, 1e-5)

    # Final results with testloader
    results = bert_utils.test(net, testloader, DEVICE, logger=LOGGER, source="Final Results") 
    print(f"Results: {results}")
    LOGGER.log_experiment()


if __name__ == "__main__":
    main()