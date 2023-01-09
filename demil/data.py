import torch
from torch.utils.data import DataLoader
from typing import Literal, Tuple, List, Union
from transformers import PreTrainedTokenizer, AutoTokenizer
from torchvision import transforms
from PIL import Image
from demil import settings
import demil.utils as utils
from demil.utils import User, TwitterUser, Post
import pandas as pd
import scipy.io as sio
import copy
import random
from datetime import datetime
from pathlib import Path
import open_clip


def shuffle_posts(
    captions: List[str],
    images: List[Union[torch.tensor, Image.Image]],
    timestamps: torch.tensor,
) -> Tuple[List[str], List[Union[torch.tensor, Image.Image]], torch.tensor]:
    padding = timestamps[len(captions):]
    timestamps = timestamps.tolist()
    lists = list(zip(captions, images, timestamps))
    random.shuffle(lists)
    captions, images, timestamps = zip(*lists)
    timestamps = torch.tensor(timestamps)
    timestamps = torch.cat([timestamps, padding])
    return list(captions), list(images), timestamps


def get_random_subseqs(
    l: List[Union[User, TwitterUser]], k: int
) -> List[List[Union[User, TwitterUser]]]:
    MIN_LEN_SEQ = 8
    if len(l) < MIN_LEN_SEQ:
        return []
    subseqs = []
    for seq_size in range(MIN_LEN_SEQ, len(l)):
        for i in range(len(l) - seq_size, -1, -1):
            seq = l[i : i + seq_size]
            subseqs.append(seq)

    n_elements = min(len(subseqs), k)
    samples = random.sample(subseqs, k=n_elements)
    return samples


def augment_data(
    dataset: List[Union[User, TwitterUser]]
) -> List[Union[User, TwitterUser]]:
    new_dataset: List[Union[User, TwitterUser]] = []
    for user in dataset:
        new_dataset.append(user)
        posts = get_random_subseqs(user.posts, k=5)
        for p in posts:
            new_user = copy.deepcopy(user)
            new_user.posts = copy.deepcopy(p)
            new_dataset.append(new_user)
    return new_dataset


def random_crop(posts: List[Union[User, TwitterUser]]) -> List[Union[User, TwitterUser]]:
    crop = get_random_subseqs(posts, 1)
    if crop:
        return crop[0]
    return crop


# class DepressBR(torch.utils.data.Dataset):
#     def __init__(
#         self,
#         dataset_type: Literal["train", "val", "test"],
#         tokenizer: PreTrainedTokenizer,
#         max_seq_length: int = settings.MAX_SEQ_LENGTH,
#         data_augmentation: bool = False,
#         get_raw_data: bool = False,
#         shuffle_posts: bool = False,
#     ):
#         self.dataset_type = dataset_type
#         self.max_seq_length = max_seq_length
#         self.shuffle_posts = shuffle_posts
#         self.dataset = utils.get_depressbr_dataset(dataset_type)
#         self.tokenizer = tokenizer
#         self.max_seq_length = max_seq_length
#         self.data_augmentation = data_augmentation
#         self.get_raw_data = get_raw_data

#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, i):
#         user = self.dataset[i]
#         texts_crop = None
#         texts = user.posts[-self.max_seq_length :]
#         if (
#             self.data_augmentation
#             and self.dataset_type == "train"
#             and random.random() < 0.20
#         ):
#             texts_crop = random_crop(texts)
#         texts = texts_crop if texts_crop else texts
#         label = user.label
#         if self.shuffle_posts:
#             random.shuffle(texts)
#         if self.get_raw_data:
#             return (texts, None, None, label)
#         captions_tensors = self.tokenizer(
#             texts,
#             add_special_tokens=True,
#             max_length=512,
#             padding="max_length",
#             return_tensors="pt",
#             return_attention_mask=True,
#             truncation=True,
#         )
#         captions_tensors["attention_mask"] = captions_tensors["attention_mask"].float()
#         return (captions_tensors, None, None, label)


# def depressbr_collate_fn(data: Tuple):
#     x = 0
#     labels = torch.zeros(
#         len(data), device=data[0][0]["input_ids"].device, dtype=torch.long
#     )
#     batch_input_ids = []
#     batch_attn_mask_ids = []
#     batch_post_attn_mask = []
#     # Here, we want to createa a Sequence [p1, ..., pn] where the first element of
#     # the sequence starts in p1 and goes until pi, which is the last element of
#     # the sequence. From pi ... pn we padd the sequence until settings.MAX_SEQ_LENGTH
#     for i, example in enumerate(data):

#         textual_data, _, _, label = example
#         input_ids, attn_mask = textual_data["input_ids"], textual_data["attention_mask"]
#         fill = settings.MAX_SEQ_LENGTH - input_ids.size(0)
#         posts_mask = torch.ones(
#             settings.MAX_SEQ_LENGTH, dtype=torch.bool, device=attn_mask.device
#         )
#         posts_mask[: input_ids.size(0)] = False
#         n_tokens = input_ids.size(1)
#         input_ids_pad = []
#         attn_mask_pad = []
#         batch_post_attn_mask.append(posts_mask)

#         input_ids_pad.append(input_ids)
#         attn_mask_pad.append(attn_mask)

#         for _ in range(0, fill):
#             aux_input_ids, aux_attn_mask = get_input_ids_attn_mask(
#                 n_tokens,
#                 device=input_ids.device,
#                 i_ids_dtype=input_ids.dtype,
#                 attn_dtype=attn_mask.dtype,
#             )
#             input_ids_pad.append(aux_input_ids)
#             attn_mask_pad.append(aux_attn_mask)

#         batch_input_ids.append(torch.cat(input_ids_pad, 0))
#         batch_attn_mask_ids.append(torch.cat(attn_mask_pad, 0))

#         labels[i] = label

#     batch_input_ids = torch.stack(batch_input_ids)
#     batch_attn_mask_ids = torch.stack(batch_attn_mask_ids)
#     batch_post_attn_mask = torch.stack(batch_post_attn_mask)

#     return (
#         (batch_input_ids, batch_attn_mask_ids),
#         None,
#         batch_post_attn_mask,
#         None,
#         labels,
#     )


class DepressionCorpus(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset: Literal["instagram", "DepressBR", "eRisk2021", "LOSADA2016", "eRisk+LOSADA", "twitter"],
        period: Literal[60, 212, 365, -1],
        dataset_type: Literal["train", "val", "test"],
        tokenizer: PreTrainedTokenizer,
        regression: bool = False,
        max_seq_length: int = settings.MAX_SEQ_LENGTH,
        data_augmentation: bool = False,
        get_raw_data: bool = False,
        shuffle_posts: bool = False,
        mil: bool = True,
        use_visual: bool = False,
        multimodal_model: str = ""
    ):
        self.dataset = get_mil_dataset(dataset, period, dataset_type) if mil else get_sil_dataset(dataset, period, dataset_type)
        self.dataset_type = dataset_type
        self.tokenizer = tokenizer
        self.regression = regression
        self.max_seq_length = max_seq_length
        self.shuffle_posts = shuffle_posts
        self.data_augmentation = data_augmentation
        self.get_raw_data = get_raw_data
        self.mil = mil
        self.use_visual = use_visual

        if multimodal_model:
            _, _, self.transform = open_clip.create_model_and_transforms(settings.HUGGINGFACE_CLIP_TO_VISUAL[multimodal_model])
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize([224, 224], interpolation=Image.LANCZOS),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

    def get_time_slots(self, tstamps: List[datetime]):
        idx = [0]
        time_dist = [
            15 * 60,
            30 * 60,
            60 * 60,
            8 * 60 * 60,
            24 * 60 * 60,
            3 * 24 * 60 * 60,
            7 * 24 * 60 * 60,
        ]
        pad_idx = len(time_dist) + 1
        for i in range(len(tstamps) - 1):
            seconds = int((tstamps[i + 1] - tstamps[i]).total_seconds())
            for j, t in enumerate(time_dist):
                if t > seconds:
                    break
            idx.append(j + 1)
        idx = torch.tensor(idx, dtype=torch.long)
        pad = pad_idx * torch.ones(self.max_seq_length - len(idx), dtype=torch.long)
        return torch.cat([idx, pad])

    def __len__(self):
        return len(self.dataset)

    def get_score(self, bdi: int) -> int:
        if self.regression:
            return bdi
        if bdi < 20:
            return 0
        else:
            return 1
    
    def getitem_mil(self, i):
        CAPTION_IDX = 0
        IMG_PATH_IDX = 1
        TIMESTAMP_IDX = 2
        user = self.dataset[i]
        posts_crop = None
        if (
            self.data_augmentation
            and self.dataset_type == "train"
            and random.random() < 0.40
        ):
            posts_crop = random_crop(user.posts)
        posts = posts_crop if posts_crop else user.posts
        score = self.get_score(user.bdi)
        images = []
        captions = []
        timestamps = []
        for i, post in enumerate(posts[::-1], 1):
            if i > self.max_seq_length:
                break
            captions.insert(0, post[CAPTION_IDX])
            if not pd.isna(post[IMG_PATH_IDX]) and self.use_visual:
                img_path = settings.PATH_TO_INSTAGRAM_DATA / post[IMG_PATH_IDX]
                image = Image.open(img_path)
                img = image.copy()
                image.close()
                if not self.get_raw_data:
                    img = self.transform(img)
                images.insert(0, img)
            timestamps.insert(0, post[TIMESTAMP_IDX].to_pydatetime())

        timestamps = self.get_time_slots(timestamps)
        if self.shuffle_posts:
            captions, images, timestamps = shuffle_posts(captions, images, timestamps)
        if self.get_raw_data:
            return (captions, images, timestamps, score)

        captions_tensors = self.tokenizer(
            captions,
            add_special_tokens=True,
            max_length=512,
            padding="max_length",
            return_tensors="pt",
            return_attention_mask=True,
            truncation=True,
        )

        captions_tensors["attention_mask"] = captions_tensors["attention_mask"].float()
        if images:
            images = torch.stack(images)
        else:
            images = torch.tensor([])
            # images = torch.zeros_like(captions_tensors["attention_mask"])

        return (captions_tensors, images, timestamps, score)
    
    def getitem_sil(self, i):
        post = self.dataset[i]
        caption, image_path, date, username, label = post
        label = self.get_score(label)
        img = None
        captions_tensors = self.tokenizer(
            caption,
            add_special_tokens=True,
            max_length=512,
            padding="max_length",
            return_tensors="pt",
            return_attention_mask=True,
            truncation=True,
        )
        captions_tensors["attention_mask"] = captions_tensors["attention_mask"].float()
        img = None
        if not pd.isna(image_path) and self.use_visual:
            img_path = settings.PATH_TO_INSTAGRAM_DATA / image_path
            image = Image.open(img_path)
            img = image.copy()
            image.close()
            if not self.get_raw_data:
                img = self.transform(img)

        if self.get_raw_data:
            return (caption, img, username, date, label)
        
        if img is None:
            img = torch.tensor([])
            # img = torch.zeros_like(captions_tensors["attention_mask"])
        
        if self.dataset_type == "test":
            return (captions_tensors["input_ids"], captions_tensors["attention_mask"], img, username, label)

        return (captions_tensors["input_ids"], captions_tensors["attention_mask"], img, "", label)

    
    def __getitem__(self, i):
        if self.mil:
            return self.getitem_mil(i)
        return self.getitem_sil(i)


def get_input_ids_attn_mask(
    n_tokens: int, device: torch.device, i_ids_dtype: torch.dtype, attn_dtype: torch.dtype
) -> Tuple[torch.Tensor, torch.Tensor]:
    if settings.LANGUAGE_MODEL in settings.BERTIMBAU or settings.LANGUAGE_MODEL in settings.ENGLISH_BERT:
        input_ids = torch.zeros(size=(1, n_tokens), device=device, dtype=i_ids_dtype)
        input_ids[0][0], input_ids[0][1] = 101, 102
        attn_mask = torch.zeros(size=(1, n_tokens), device=device, dtype=attn_dtype)
        attn_mask[0][0], attn_mask[0][1] = 1.0, 1.0

    elif settings.LANGUAGE_MODEL in settings.XLM or settings.LANGUAGE_MODEL in settings.CLIP:
        input_ids = torch.ones(size=(1, n_tokens), device=device, dtype=i_ids_dtype)
        input_ids[0][0], input_ids[0][1] = 0, 2
        attn_mask = torch.zeros(size=(1, n_tokens), device=device, dtype=attn_dtype)
        attn_mask[0][0], attn_mask[0][1] = 1.0, 1.0
    else:
        raise ValueError(
            f"Language model '{settings.LANGUAGE_MODEL}' is not currently supported."
        )

    return input_ids, attn_mask


# data is a list containing the batch examples, where each element is an example.
#  data[i] is a list with size 3, where index 0 is the textual data, index 1 is
#  visual data, and index 2 is the label
def collate_fn_mil(data: Tuple):
    x = 0
    labels = torch.zeros(len(data), device=data[0][1].device, dtype=torch.long)
    batch_input_ids = []
    batch_attn_mask_ids = []
    batch_visual_data = []
    batch_post_attn_mask = []
    timestamps = []
    # Here, we want to createa a Sequence [p1, ..., pn] where the first element of
    # the sequence starts in p1 and goes until pi, which is the last element of
    # the sequence. From pi ... pn we padd the sequence until settings.MAX_SEQ_LENGTH
    for i, example in enumerate(data):

        textual_data, visual_data, timestamp, label = example
        timestamps.append(timestamp)
        input_ids, attn_mask = textual_data["input_ids"], textual_data["attention_mask"]
        fill = settings.MAX_SEQ_LENGTH - input_ids.size(0)
        posts_mask = torch.ones(
            settings.MAX_SEQ_LENGTH, dtype=torch.bool, device=attn_mask.device
        )
        posts_mask[: input_ids.size(0)] = False
        n_tokens = input_ids.size(1)
        input_ids_pad = []
        attn_mask_pad = []

        if len(visual_data) != 0:
            vis_padding = torch.zeros(size=[fill, *visual_data.size()[1:]])
            batch_visual_data.append(torch.cat([visual_data, vis_padding], 0))
        else:
            batch_visual_data.append(torch.tensor([]))
        batch_post_attn_mask.append(posts_mask)

        input_ids_pad.append(input_ids)
        attn_mask_pad.append(attn_mask)

        for _ in range(0, fill):
            aux_input_ids, aux_attn_mask = get_input_ids_attn_mask(
                n_tokens,
                device=input_ids.device,
                i_ids_dtype=input_ids.dtype,
                attn_dtype=attn_mask.dtype,
            )
            input_ids_pad.append(aux_input_ids)
            attn_mask_pad.append(aux_attn_mask)

        batch_input_ids.append(torch.cat(input_ids_pad, 0))
        batch_attn_mask_ids.append(torch.cat(attn_mask_pad, 0))

        labels[i] = label

    batch_input_ids = torch.stack(batch_input_ids)
    batch_attn_mask_ids = torch.stack(batch_attn_mask_ids)
    batch_visual_data = torch.stack(batch_visual_data)
    batch_post_attn_mask = torch.stack(batch_post_attn_mask)

    return (
        (batch_input_ids, batch_attn_mask_ids),
        batch_visual_data,
        batch_post_attn_mask,
        torch.stack(timestamps),
        labels,
    )

def get_mil_dataset(
    dataset: Literal["instagram", "DepressBR", "eRisk2021", "LOSADA2016", "eRisk+LOSADA", "twitter"], 
    period: Literal[60, 212, 365, -1],
    split: Literal["train", "val", "test"]
) -> List[User]:
    data = utils.load_dataframes(dataset, period, split)
    data = utils.get_users_info(utils.sort_by_date(data))
    return data

def get_sil_dataset(
    dataset: Literal["instagram", "DepressBR", "eRisk2021", "LOSADA2016", "eRisk+LOSADA", "twitter"], 
    period: Literal[60, 212, 365, -1],
    split: Literal["train", "val", "test"]
) -> List[Post]:
    data = utils.load_dataframes(dataset, period, split)
    data = utils.get_posts_info(data)
    return data


def get_dataloaders(
    period: Literal[60, 212, 365, -1],
    batch_size: int,
    collate_fn,
    shuffle: bool,
    tokenizer: PreTrainedTokenizer = None,
    regression: bool = False,
    augment_data: bool = False,
    dataset: Literal["instagram", "DepressBR", "eRisk2021", "LOSADA2016", "eRisk+LOSADA", "twitter"] = "instagram",
    shuffle_posts: bool = False,
    mil: bool = True,
    use_visual: bool = False,
    multimodal_model: str = ""
):
    # available_datasets = ["instagram", "DepressBR", "eRisk2021", "LOSADA2016", "eRisk+LOSADA", "twitter"]
    # if dataset not in available_datasets:
    #     raise ValueError(f"Dataset is not one of the following: {available_datasets}")
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(settings.LANGUAGE_MODEL)

    # if dataset == "DepressBR":
    #     train = DepressBR("train", tokenizer, data_augmentation=augment_data)
    #     val = DepressBR("val", tokenizer, data_augmentation=augment_data)
    #     test = DepressBR("test", tokenizer, data_augmentation=augment_data)
    #     collate = depressbr_collate_fn
    # else:
    train = DepressionCorpus(dataset, period, "train", tokenizer, regression, data_augmentation=augment_data, shuffle_posts=shuffle_posts, mil=mil, use_visual=use_visual, multimodal_model=multimodal_model)
    val = DepressionCorpus(dataset, period, "val", tokenizer, regression, data_augmentation=augment_data, mil=mil, use_visual=use_visual, multimodal_model=multimodal_model)
    test = DepressionCorpus(dataset, period, "test", tokenizer, regression, data_augmentation=augment_data, mil=mil, use_visual=use_visual, multimodal_model=multimodal_model)
    collate = collate_fn

    train_loader = DataLoader(
        train,
        shuffle=shuffle,
        batch_size=batch_size,
        num_workers=settings.WORKERS,
        pin_memory=True,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val,
        batch_size=batch_size,
        num_workers=settings.WORKERS,
        pin_memory=True,
        collate_fn=collate,
    )
    test_loader = DataLoader(
        test,
        batch_size=batch_size,
        num_workers=settings.WORKERS,
        pin_memory=True,
        collate_fn=collate,
    )
    return train_loader, val_loader, test_loader