'''
    Code modified from https://github.com/adap/flower/blob/d1eb90f74714a9c10ddbeefb767b56be7b61303d/examples/quickstart_huggingface/client.py#L11
'''
import argparse

import pandas as pd

from collections import OrderedDict
import warnings

import flwr as fl
import torch
import numpy as np

import random
from torch.utils.data import DataLoader

from datasets import load_dataset, load_metric, Dataset

from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification
from transformers import AdamW

warnings.filterwarnings("ignore", category=UserWarning)

def get_args():
    parser = argparse.ArgumentParser(description="Creating a bert client for federated learning")
    parser.add_argument("--gpu", type=int, default=7, help="wich device this client will use")
    parser.add_argument("--dataset_piece", type=int, default=0, help="wich subdivision of the dataset this client will have")

    args = parser.parse_args()
    return args

args = get_args()
DEVICE = torch.device(f"cuda:{args.gpu}")
CHECKPOINT = "arthurbittencourt/bert_classifier_2e16"
DATASET_DIR = "/home/arthurbittencourt/depression-demil/demil/data/eRisk2021_partitioned/"

def dataset_prep(ds):
    ds.bdi[ds.bdi < 20] = 0
    ds.bdi[ds.bdi >= 20] = 1

    ds['text'] = ds.caption.fillna('')
    ds['text'] = ds.text.str.replace(r"http\S+", "HTTP")
    ds['text'] = ds.text.str.replace(r"@\w*", "USER")

    ds = ds.rename(columns={'bdi':'label'})

    ds = ds[['text', 'label']]

    return ds

def load_data(piece):
    """Load eRisk2021 data (training and eval)"""
    df_train = pd.read_csv(DATASET_DIR + f'train/{piece}.csv')
    df_test = pd.read_csv(DATASET_DIR + f'test/{piece}.csv')
    df_val = pd.read_csv(DATASET_DIR + f'val/{piece}.csv')

    df_train = dataset_prep(df_train)
    df_test = dataset_prep(df_test)
    df_val = dataset_prep(df_val)

    ds_train = Dataset.from_pandas(df_train)
    ds_test = Dataset.from_pandas(df_test)
    ds_val = Dataset.from_pandas(df_val)

    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True)

    ds_train = ds_train.map(tokenize_function, batched=True)
    ds_test = ds_test.map(tokenize_function, batched=True)
    ds_val = ds_val.map(tokenize_function, batched=True)

    ds_train = ds_train.remove_columns("text")
    ds_test = ds_test.remove_columns("text")
    ds_val = ds_val.remove_columns("text")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    BATCH_SIZE = 6
    
    trainloader = DataLoader(
        ds_train,
        shuffle=True,
        batch_size=BATCH_SIZE,
        collate_fn=data_collator,
    )

    testloader = DataLoader(
        ds_test, 
        batch_size=BATCH_SIZE, 
        collate_fn=data_collator
    )

    valloader = DataLoader(
        ds_val, 
        batch_size=BATCH_SIZE, 
        collate_fn=data_collator
    )

    return trainloader, testloader, valloader


def train(net, trainloader, epochs):
    optimizer = AdamW(net.parameters(), lr=5e-5)
    net.train()
    for _ in range(epochs):
        for batch in trainloader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = net(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


def test(net, testloader):
    accuracy = load_metric("accuracy")
    precision = load_metric("precision")
    recall = load_metric("recall")
    f1 = load_metric("f1")

    loss = 0
    net.eval()

    for batch in testloader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        with torch.no_grad():
            outputs = net(**batch)
        logits = outputs.logits
        loss += outputs.loss.item()
        predictions = torch.argmax(logits, dim=-1)

        accuracy.add_batch(predictions=predictions, references=batch["labels"])
        precision.add_batch(predictions=predictions, references=batch["labels"])
        recall.add_batch(predictions=predictions, references=batch["labels"])
        f1.add_batch(predictions=predictions, references=batch["labels"])


    testloader.dataset
    loss /= len(testloader.dataset)
    metrics = {
        "accuracy":accuracy.compute()["accuracy"],
        "precision":precision.compute()["precision"],
        "recall":recall.compute()["recall"],
        "f1":f1.compute()["f1"],
    }
    return loss, metrics

def main():

    net = AutoModelForSequenceClassification.from_pretrained(
        CHECKPOINT, num_labels=2
    ).to(DEVICE)

    trainloader, testloader, valloader = load_data(args.dataset_piece)

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
            train(net, trainloader, epochs=1)
            print("Training Finished.")
            return self.get_parameters(config={}), len(trainloader), {}

        def evaluate(self, parameters, config):
                self.set_parameters(parameters)
                loss, metrics = test(net, valloader)
                accuracy = float(metrics["accuracy"])
                precision = float(metrics["precision"])
                recall = float(metrics["recall"])
                f1 = float(metrics["f1"])
                print("Metrics: ", metrics)
                return float(loss), len(valloader), {
                    "accuracy": accuracy, 
                    "precision": precision, 
                    "recall": recall, 
                    "fscore": f1
                }

    # Start client
    fl.client.start_numpy_client(server_address="[::]:8080", client=BertClassifierClient(), grpc_max_message_length=1543194403)
    # loss, metrics = test(net=net, testloader=valloader)
    # print("Results: ")
    # print(metrics)

    '''
    print(f"WRITING WEIGHTS FOR SEED {seed}")
    weights = [val.cpu().numpy() for _, val in model.state_dict().items()]
    with open("weights.bin", "wb") as handle:
        pk.dump(weights, handle, protocol=pk.HIGHEST_PROTOCOL)
    '''


if __name__ == "__main__":
    main()