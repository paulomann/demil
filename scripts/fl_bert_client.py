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
from transformers import AdamW
from evaluate import load

import bert_utils

warnings.filterwarnings("ignore", category=UserWarning)

def get_args():
    parser = argparse.ArgumentParser(description="Creating a bert client for federated learning")
    parser.add_argument("--gpu", type=int, default=7, help="wich device this client will use")
    parser.add_argument("--dataset_piece", type=int, default=0, help="wich subdivision of the dataset this client will have")
    parser.add_argument("--checkpoint", type=str, default="bert-base-uncased", help="what checkpoint to use, leave blank for bert-base-uncased")
    parser.add_argument("--epochs", type=int, default=1, help="how many epochs to run on this client")

    args = parser.parse_args()
    return args

args = get_args()

DEVICE = torch.device(f"cuda:{args.gpu}")
CHECKPOINT = args.checkpoint
EPOCHS = args.epochs

def train(net, trainloader, epochs):
    optimizer = AdamW(net.parameters(), lr=1e-5)
    net.train()
    for _ in range(epochs):
        for batch in trainloader:
            if 'username' in batch: batch.pop('username')
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = net(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

def main():

    net = AutoModelForSequenceClassification.from_pretrained(
        CHECKPOINT, num_labels=2
    ).to(DEVICE)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    trainloader, _, valloader = bert_utils.load_data(partition=args.dataset_piece, tokenizer=tokenizer, batch_size=6)

    # Flower client
    class BertClassifierClient(fl.client.NumPyClient):
        def get_parameters(self, config=None):
            return [val.cpu().numpy() for _, val in net.state_dict().items()]

        def set_parameters(self, parameters):
            params_dict = zip(net.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
            net.load_state_dict(state_dict, strict=True)

        def fit(self, parameters, config):
            self.set_parameters(parameters)
            print("Training Started...")
            train(net, trainloader, epochs=EPOCHS)
            print("Training Finished.")
            return self.get_parameters(config={}), len(trainloader), {}

        def evaluate(self, parameters, config):
                self.set_parameters(parameters)
                metrics = bert_utils.test(net=net, dataloader=valloader, device=DEVICE, logger=None, source=f'Client {args.dataset_piece}')
                
                loss = metrics['loss']

                print("Metrics: ", metrics)
                
                return float(loss), len(valloader), metrics

                '''
                float(loss), len(valloader), {
                    "source": f"Client {args.dataset_piece}",
                    "accuracy": accuracy, 
                    "precision": precision, 
                    "recall": recall, 
                    "fscore": f1,
                    "loss": loss
                }
                '''

    
    # Start client
    fl.client.start_numpy_client(server_address="[::]:8080", client=BertClassifierClient(), grpc_max_message_length=1543194403)

    '''
    print(f"WRITING WEIGHTS FOR SEED {seed}")
    weights = [val.cpu().numpy() for _, val in model.state_dict().items()]
    with open("weights.bin", "wb") as handle:
        pk.dump(weights, handle, protocol=pk.HIGHEST_PROTOCOL)
    '''


if __name__ == "__main__":
    main()