'''
    Code modified from https://github.com/adap/flower/blob/d1eb90f74714a9c10ddbeefb767b56be7b61303d/examples/quickstart_huggingface/client.py#L11
'''
import pandas as pd
import os.path

import warnings

import flwr as fl
import torch
import numpy as np

from torch.utils.data import DataLoader
from datasets import Dataset
from evaluate import load
from transformers import AutoTokenizer, DataCollatorWithPadding

from sklearn.metrics import precision_recall_fscore_support

import time

warnings.filterwarnings("ignore", category=UserWarning)

# CHECKPOINT = "mental/mental-bert-base-uncased"
DATASET_DIR = "/home/arthurbittencourt/depression-demil/demil/data/"
RESULTS_DIR = "/home/arthurbittencourt/depression-demil/demil/results/"

def dataset_prep(df:pd.DataFrame):
    ds = df.copy(deep=True)
    ds.bdi = ds.bdi.where(ds.bdi >= 20, 0)
    ds.bdi = ds.bdi.where(ds.bdi < 20, 1)
    ds = ds.dropna(axis=0, subset=["caption"])
    ds = ds.reset_index()

    ds['text'] = ds.caption.str.replace(r"http\S+", "HTTP")
    ds['text'] = ds.text.str.replace(r"@\w*", "USER")
    ds['username'] = ds.username.str.replace("erisk2021-T3_Subject", "")
    ds['username'] = ds.username.astype(int)

    ds = ds.rename(columns={'bdi':'label'})

    ds = ds[['username', 'text', 'label']]
    #ds = ds[['text', 'label']]

    return ds

def get_user_dict(df):
    user_dict = {user:{'negative':0, 'positive':0, 'bdi':df[df.username==user].bdi.unique()[0]} for user in df.username.unique()}



def load_data(dataset="eRisk2021", partition=False, tokenizer=AutoTokenizer.from_pretrained("bert-base-uncased"), batch_size=6):

    if partition != False:
        df_train = pd.read_csv(DATASET_DIR + dataset + '_partitioned/' +'train/' + f'{partition}.csv')
        df_test = pd.read_csv(DATASET_DIR + dataset + '_partitioned/' +'/test/' + f'{partition}.csv')
        df_val = pd.read_csv(DATASET_DIR + dataset + '_partitioned/' +'/val/' + f'{partition}.csv')
        
    else:
        df_train = pd.read_csv(DATASET_DIR + dataset + '/' + f'train.csv')
        df_test = pd.read_csv(DATASET_DIR + dataset + '/' + f'test.csv')
        df_val = pd.read_csv(DATASET_DIR + dataset + '/' + f'val.csv')

    df_train = dataset_prep(df_train)
    ds_train = Dataset.from_pandas(df_train)
    
    df_test = dataset_prep(df_test)
    ds_test = Dataset.from_pandas(df_test)

    df_val = dataset_prep(df_val)
    ds_val = Dataset.from_pandas(df_val)

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True)

    ds_train = ds_train.map(tokenize_function, batched=True)
    ds_test = ds_test.map(tokenize_function, batched=True)
    ds_val = ds_val.map(tokenize_function, batched=True)

    ds_train = ds_train.remove_columns("text")
    ds_test = ds_test.remove_columns("text")
    ds_val = ds_val.remove_columns("text")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    BATCH_SIZE = batch_size

    trainloader = DataLoader(
        ds_train, 
        batch_size=BATCH_SIZE, 
        collate_fn=data_collator
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

class UserMetrics():
    def __init__(self):
        self.users = {}
    
    def add_batch(self, user_ids, predictions, references):
        for i in range(len(user_ids)):
            id = user_ids[i].item()
            pred = predictions[i].item()
            ref = references[i].item()
            
            if id not in self.users:
                self.users[id] = {'positive':pred, 'total':1, 'label':ref}

            else:
                self.users[id]['positive'] += pred
                self.users[id]['total'] += 1

    def compute(self):        
        pred = []
        label = []
        
        print("==============[Computing UserMetrics]=================")
        print("Users: ", self.users)
        for user in self.users:
            positive = self.users[user]['positive']
            total = self.users[user]['total']

            if positive/total > 0.5: pred.append(1) # se a quantidade de posts positivos for maior que a metade, considere o usuário como positivo
            else: pred.append(0)

            label.append(self.users[user]['label'])

        print("Predictions:", pred)
        print("Labels:", label)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true=label, y_pred=pred, average='macro')


        return {'user_precision':precision, 'user_recall':recall, 'user_f1':f1}
        


def test(net, dataloader, device, logger = None, source='Aggregate Model'):
    
    #print("==============[Evaluating]=================")
    accuracy = load("accuracy")
    precision = load("precision")
    recall = load("recall")
    f1 = load("f1")
    um = UserMetrics()

    loss = 0
    net.eval()

    preds = []
    labels = []
    with torch.no_grad():
        for batch in dataloader:
            if 'username' in batch:
                usernames = batch.pop('username')
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = net(**batch)
            logits = outputs.logits
            loss += outputs.loss.item()
            predictions = torch.argmax(logits, dim=-1)

            preds.extend(predictions.tolist())
            labels.extend(batch["labels"].tolist())

            accuracy.add_batch(predictions=predictions, references=batch["labels"])
            precision.add_batch(predictions=predictions, references=batch["labels"])
            recall.add_batch(predictions=predictions, references=batch["labels"])
            f1.add_batch(predictions=predictions, references=batch["labels"])
            um.add_batch(usernames, predictions, batch['labels'])

    print("Preds: ", preds, '\n Labels: ', labels)    
    #print(f"Predictions: \n Negative: {positive_preds}\n Positive: {negative_preds}")

    dataloader.dataset
    loss /= len(dataloader.dataset)
    user_metrics = um.compute()
    metrics = {
        "source": source,
        "accuracy": accuracy.compute()["accuracy"],
        "precision": precision.compute()["precision"],
        "recall": recall.compute()["recall"],
        "fscore": f1.compute()["f1"],
        "loss": loss,
        "user_precision":user_metrics["user_precision"],
        "user_recall":user_metrics["user_recall"],
        "user_f1":user_metrics["user_f1"],
        "timestamp":time.time()
    }

    if logger:
        #print("[INFO] Appending metrics from test")
        logger.append_metrics(metrics)

    return metrics

    

def round_eval(net, device, logger, eval_msg = "Final Evaluation"): # Used for testing aggregated bert models, after a fl round
    print(f"===============[{eval_msg}]===============")
    
    _, testloader, _ = load_data()
    metrics = test(net, testloader, device)

    logger.append_metrics(metrics)
    
    print(metrics)
    print(f"===============[{eval_msg}]===============")
    
    return metrics

class ExperimentLogger():
    def __init__(self, exp_name: str):
        self.exp_name = exp_name
        self.metric_rows = []
    
    def append_metrics(self, metrics):
        self.metric_rows.append(metrics)

    def log_experiment(self):
        df = pd.DataFrame(self.metric_rows)
        df.to_csv(RESULTS_DIR + self.exp_name + ".csv", decimal=",")


def append_metrics_to_csv(experiment_name, metrics):
    
    df = pd.DataFrame(metrics, index=[0])
    print("Dataframe:", df)
    df = df[["source", "loss", "accuracy", "precision", "recall", "fscore"]]

    if not experiment_name.endswith(".csv"): experiment_name += ".csv"
    csv_dir = RESULTS_DIR + experiment_name

    df.to_csv(csv_dir, decimal=",", mode='a', header=(not os.path.exists(csv_dir)))
    
    


if __name__ == "__main__":
    from transformers import AutoModelForSequenceClassification
    net = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).to(7)
    trainloader, testloader, valloader = load_data()
    results = test(net=net, dataloader=testloader, device=7)

    logger = ExperimentLogger("baseline_evaluation")
    logger.append_metrics(results)
    logger.log_experiment()

    print(results)
