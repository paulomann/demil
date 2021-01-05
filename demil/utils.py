import pandas as pd
import numpy as np
from demil import settings
from typing import Literal, Tuple, Dict, List
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
from transformers import PreTrainedTokenizer, AutoTokenizer
from demil import settings


@dataclass
class User:
    username: str
    bdi: int
    posts: List[Tuple[str, Path, datetime]]


def load_dataframes(
    period: Literal[60, 212, 365], dataset: Literal["train", "val", "test"]
) -> pd.DataFrame:
    if dataset == "train":
        data = pd.read_csv(settings.DEPRESSION_CORPUS / f"{period}" / "train.csv")
    elif dataset == "val":
        data = pd.read_csv(settings.DEPRESSION_CORPUS / f"{period}" / "val.csv")
    elif dataset == "test":
        data = pd.read_csv(settings.DEPRESSION_CORPUS / f"{period}" / "test.csv")
    else:
        raise ValueError(f"The {dataset=} parameter does not exist.")

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