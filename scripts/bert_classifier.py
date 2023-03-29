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

warnings.filterwarnings("ignore", category=UserWarning)

def get_args():
    parser = argparse.ArgumentParser(description="Creating a bert client for federated learning")
    parser.add_argument("--gpu", type=int, default=7, help="wich device this client will use")
    parser.add_argument("--checkpoint", type=str, default="bert-base-uncased", help="what checkpoint to use, leave blank for bert-base-uncased")
    parser.add_argument("--epochs", type=int, default=1, help="how many epochs to run on this client")

    args = parser.parse_args()
    return args

args = get_args()

DEVICE = torch.device(f"cuda:{args.gpu}")
CHECKPOINT = args.checkpoint
EPOCHS = args.epochs
CSV_SAVE_DIR = f'/home/arthurbittencourt/depression-demil/demil/results/centralized_model.csv' 

DATASET_DIR = "/home/arthurbittencourt/depression-demil/demil/data/eRisk2021/"

def dataset_prep(ds):
    ds.bdi[ds.bdi < 20] = 0
    ds.bdi[ds.bdi >= 20] = 1

    ds['text'] = ds.caption.fillna('')
    ds['text'] = ds.text.str.replace(r"http\S+", "HTTP")
    ds['text'] = ds.text.str.replace(r"@\w*", "USER")

    ds = ds.rename(columns={'bdi':'label'})

    ds = ds[['text', 'label']]

    return ds

def load_data():
    """Load eRisk2021 data (training and eval)"""
    df_train = pd.read_csv(DATASET_DIR + f'train.csv')
    df_test = pd.read_csv(DATASET_DIR + f'test.csv')
    df_val = pd.read_csv(DATASET_DIR + f'val.csv')

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
    optimizer = AdamW(net.parameters(), lr=1e-5)
    net.train()
    for i in range(epochs):
        for batch in trainloader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = net(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print(f"Round {i+1} Loss: {loss}")


def test(net, valloader):
    accuracy = load("accuracy")
    precision = load("precision")
    recall = load("recall")
    f1 = load("f1")

    loss = 0
    net.eval()

    for batch in valloader:
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


    valloader.dataset
    loss /= len(valloader.dataset)
    metrics = {
        "accuracy":accuracy.compute()["accuracy"],
        "precision":precision.compute()["precision"],
        "recall":recall.compute()["recall"],
        "f1":f1.compute()["f1"],
    }
    return loss, metrics
    
def append_result(loss, metrics):
    df = pd.DataFrame(metrics)
    df['loss'] = loss
    df.to_csv(CSV_SAVE_DIR, decimal=',', mode='a', header=(not os.path.exists(CSV_SAVE_DIR)))

def main():

    net = AutoModelForSequenceClassification.from_pretrained(
        CHECKPOINT, num_labels=2
    ).to(DEVICE)

    trainloader, _, valloader = load_data()

    
    # Start Training
    print("Sarting Training...'")
    train(net, trainloader, EPOCHS)
    
    results = test(net, valloader)
    print("Results:", results)

    '''
    print(f"WRITING WEIGHTS FOR SEED {seed}")
    weights = [val.cpu().numpy() for _, val in model.state_dict().items()]
    with open("weights.bin", "wb") as handle:
        pk.dump(weights, handle, protocol=pk.HIGHEST_PROTOCOL)
    '''


if __name__ == "__main__":
    main()