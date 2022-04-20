import pandas as pd
import numpy as np
from demil import settings
from typing import Literal, Tuple, Dict, List
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
import json
from transformers import PreTrainedTokenizer, AutoTokenizer
from demil import settings
from demil.transformer import UniterConfig
import torch


@dataclass
class User:
    username: str
    bdi: int
    posts: List[Tuple[str, Path, datetime]]


@dataclass
class TwitterUser:
    username: str
    label: int
    posts: List[str]


def load_depressbr_dataframe(dataset: Literal["train", "val", "test"]) -> pd.DataFrame:
    if dataset not in ["train", "val", "test"]:
        raise ValueError(f"The {dataset=} parameter does not exist.")
    return pd.read_csv(settings.PATH_TO_DEPRESSBR / f"{dataset}.csv", sep=",")


def get_depressbr_info(df: pd.DataFrame) -> List[TwitterUser]:
    users = {}
    users_list = []
    for row in df.itertuples():
        username = row.User_ID
        text = row.Text
        label = 0 if row.Class == "no" else 1
        if username in users:
            users[username]["posts"].append(text)
        else:
            users[username] = {"label": label, "posts": []}
    for k, v in users.items():
        if len(v["posts"]) > 0:
            user = TwitterUser(k, v["label"], v["posts"])
            users_list.append(user)
    return users_list


def get_depressbr_dataset(dataset: Literal["train", "val", "test"]) -> List[TwitterUser]:
    df = load_depressbr_dataframe(dataset)
    return get_depressbr_info(df)


def load_dataframes(
    dataset: Literal["DeprUFF", "DepressBR", "eRisk2021", "LOSADA2016", "eRisk+LOSADA", "twitter"],
    period: Literal[60, 212, 365, -1],
    split: Literal["train", "val", "test"]
) -> pd.DataFrame:
    if period != -1:
        if split in ["train", "val", "test"]:
            data = pd.read_csv(settings.DEPRESSION_CORPUS / f"{period}" / f"{split}.csv")
        else:
            raise ValueError(f"The {split=} parameter does not exist.")
    else:
        data = pd.read_csv(Path(settings.DATA_PATH, dataset, f"{split}.csv"))

    return data


def sort_by_date(df: pd.DataFrame) -> pd.DataFrame:
    df.date = pd.to_datetime(df.date)
    return df.sort_values(by="date")


def get_users_info(df: pd.DataFrame) -> List[User]:
    users = {}
    users_list = []
    for row in df.iterrows():
        row = row[1]
        caption = row.caption
        username = row.username
        bdi = row.bdi
        image_path = row.img
        date = row.date
        if username in users:
            if caption is np.nan:
                caption = ""
            users[username]["posts"].append((caption, image_path, date))
        else:
            users[username] = {"bdi": bdi, "posts": []}
    for k, v in users.items():
        if len(v["posts"]) > 0:
            user = User(k, v["bdi"], v["posts"])
            users_list.append(user)
    return users_list


def get_dataset(
    period: Literal[60, 212, 365], dataset: Literal["train", "val", "test"]
) -> List[User]:
    data = load_dataframes(period, dataset)
    data = get_users_info(sort_by_date(data))
    return data


def get_statistics_for_posts():
    periods = [60, 212, 365]
    for p in periods:
        qty = []
        train, val, test = (
            get_dataset(p, "train"),
            get_dataset(p, "val"),
            get_dataset(p, "test"),
        )
        print()
        all_users = []
        all_users.extend(train)
        all_users.extend(val)
        all_users.extend(test)
        for user in all_users:
            qty.append(len(user.posts))
        print(
            f"""==== Report ({p}) ====\nAverage: {np.mean(qty)}\nStandard Deviation: {np.std(qty)}\n# Users: {len(all_users)}\nMax: {np.max(qty)}"""
        )


# def parse_with_config(parser):
#     args = parser.parse_args()
#     if args.config is not None:
#         config_args = json.load(open(args.config))
#         override_keys = {arg[2:].split('=')[0] for arg in sys.argv[1:]
#                          if arg.startswith('--')}
#         for k, v in config_args.items():
#             if k not in override_keys:
#                 setattr(args, k, v)
#     del args.config
#     return args


def get_bert_config(path: Path = settings.PATH_TO_BERT_CONFIG):
    if path is not None:
        config = UniterConfig(path)
    return config