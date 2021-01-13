import torch
from torch.utils.data import DataLoader
from typing import Literal, Tuple, List
from transformers import PreTrainedTokenizer, AutoTokenizer
from torchvision import transforms
from PIL import Image
from demil import settings
import demil.utils as utils
from demil.utils import User
import pandas as pd
import scipy.io as sio


class AVECCorpus(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_type: Literal["train", "val", "test"],
        max_seq_length: int = -1,
    ):
        self.dataset_type = dataset_type
        self.max_seq_length = max_seq_length
        self.audio, self.video, self.labels, self.size = self.load_datasets(
            self.dataset_type
        )

    def __len__(self):
        return self.size

    def load_datasets(
        self, dataset_type
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor, int]:
        metadata = pd.read_csv(settings.AVEC_METADATA, encoding=settings.encoding)

        if dataset_type == "test":
            audio_files = list(settings.AVEC_AUDIO.rglob("test*_densenet*"))
            video_files = list(settings.AVEC_VIDEO.rglob("test*_ResNet*"))
        elif dataset_type == "val":
            audio_files = list(settings.AVEC_AUDIO.rglob("development*_densenet*"))
            video_files = list(settings.AVEC_VIDEO.rglob("development*_ResNet*"))
        elif dataset_type == "train":
            audio_files = list(settings.AVEC_AUDIO.rglob("training*_densenet*"))
            video_files = list(settings.AVEC_VIDEO.rglob("training*_ResNet*"))
        else:
            raise ValueError(f"There is no split called {dataset_type}")

        video_files = sorted(video_files)
        audio_files = sorted(audio_files)

        if dataset_type != "test":
            labels = []
            for path in video_files:
                name = "_".join(path.name.split("_")[:2])
                labels.append(
                    metadata.loc[metadata["Participant_ID"] == name]["PHQ_Binary"].values[
                        0
                    ]
                )
            labels = torch.tensor(labels).long()
            assert len(video_files) == len(audio_files) == len(labels)
        else:
            labels = torch.tensor([]).long()
            assert len(video_files) == len(audio_files)

        data_size = len(video_files)
        audio_data = []
        video_data = []
        for audio_file, video_file in zip(audio_files, video_files):
            csv = pd.read_csv(audio_file, encoding=settings.encoding)
            audio_embeddings = torch.from_numpy(csv.to_numpy()[:, 2:].astype("float32"))
            audio_data.append(audio_embeddings)
            video_embeddings = torch.from_numpy(
                sio.loadmat(video_file)["feature"].astype("float32")
            )
            video_data.append(video_embeddings)

        return audio_data, video_data, labels, data_size

    def __getitem__(self, i):
        return (
            self.audio[i][: self.max_seq_length, :],
            self.video[i][: self.max_seq_length, :],
            self.labels[i] if self.dataset_type != "test" else self.labels,
        )


class DepressionCorpus(torch.utils.data.Dataset):
    def __init__(
        self,
        period: Literal[60, 212, 365],
        dataset_type: Literal["train", "val", "test"],
        tokenizer: PreTrainedTokenizer,
        regression: bool = False,
        max_seq_length: int = settings.MAX_SEQ_LENGTH,
    ):
        self.dataset = get_dataset(period, dataset_type)
        self.dataset_type = dataset_type
        self.tokenizer = tokenizer
        self.regression = regression
        self.max_seq_length = max_seq_length
        self.transform = transforms.Compose(
            [
                transforms.Resize([224, 224], interpolation=Image.LANCZOS),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __len__(self):
        return len(self.dataset)

    def get_score(self, bdi: int) -> int:
        if self.regression:
            return bdi
        if bdi < 20:
            return 0
        else:
            return 1

    def __getitem__(self, i):
        CAPTION_IDX = 0
        IMG_PATH_IDX = 1
        TIMESTAMP_IDX = 2
        user = self.dataset[i]
        posts = user.posts
        score = self.get_score(user.bdi)
        images = []
        captions = []
        for i, post in enumerate(posts[::-1], 1):
            if i > self.max_seq_length:
                break
            captions.insert(0, post[CAPTION_IDX])
            img_path = settings.PATH_TO_INSTAGRAM_DATA / post[IMG_PATH_IDX]
            image = Image.open(img_path)
            img = image.copy()
            image.close()
            img = self.transform(img)
            images.insert(0, img)

        captions_tensors = self.tokenizer(
            captions,
            add_special_tokens=True,
            max_length=512,
            padding="max_length",
            return_tensors="pt",
            return_attention_mask=True,
            truncation=True,
        )
        images = torch.stack(images)
        captions_tensors["attention_mask"] = captions_tensors["attention_mask"].float()

        return (captions_tensors, images, score)


def get_input_ids_attn_mask(
    n_tokens: int, device: torch.device, i_ids_dtype: torch.dtype, attn_dtype: torch.dtype
) -> Tuple[torch.Tensor, torch.Tensor]:
    if settings.LANGUAGE_MODEL in settings.BERTIMBAU:
        input_ids = torch.zeros(size=(1, n_tokens), device=device, dtype=i_ids_dtype)
        input_ids[0][0], input_ids[0][1] = 101, 102
        attn_mask = torch.zeros(size=(1, n_tokens), device=device, dtype=attn_dtype)
        attn_mask[0][0], attn_mask[0][1] = 1.0, 1.0

    elif settings.LANGUAGE_MODEL in settings.XLM:
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
def collate_fn(data: Tuple):
    x = 0
    labels = torch.zeros(len(data), device=data[0][1].device, dtype=torch.long)
    batch_input_ids = []
    batch_attn_mask_ids = []
    batch_visual_data = []
    batch_post_attn_mask = []
    # Here, we want to createa a Sequence [p1, ..., pn] where the first element of
    # the sequence starts in p1 and goes until pi, which is the last element of
    # the sequence. From pi ... pn we padd the sequence until settings.MAX_SEQ_LENGTH
    for i, example in enumerate(data):

        textual_data, visual_data, label = example
        input_ids, attn_mask = textual_data["input_ids"], textual_data["attention_mask"]
        fill = settings.MAX_SEQ_LENGTH - input_ids.size(0)
        posts_mask = torch.ones(
            settings.MAX_SEQ_LENGTH, dtype=torch.bool, device=attn_mask.device
        )
        posts_mask[:input_ids.size(0)] = False
        n_tokens = input_ids.size(1)
        input_ids_pad = []
        attn_mask_pad = []

        vis_padding = torch.zeros(size=[fill, *visual_data.size()[1:]])
        batch_visual_data.append(torch.cat([visual_data, vis_padding], 0))
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
        labels,
    )


def get_dataset(
    period: Literal[60, 212, 365], dataset: Literal["train", "val", "test"]
) -> List[User]:
    data = utils.load_dataframes(period, dataset)
    data = utils.get_users_info(utils.sort_by_date(data))
    return data


def get_dataloaders(
    period: Literal[60, 212, 365],
    batch_size: int,
    collate_fn,
    shuffle: bool,
    tokenizer: PreTrainedTokenizer = None,
    regression: bool = False,
):
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(settings.LANGUAGE_MODEL)
    train = DepressionCorpus(period, "train", tokenizer, regression)
    val = DepressionCorpus(period, "val", tokenizer, regression)
    test = DepressionCorpus(period, "test", tokenizer, regression)
    train_loader = DataLoader(
        train,
        shuffle=shuffle,
        batch_size=batch_size,
        num_workers=settings.WORKERS,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val,
        batch_size=batch_size,
        num_workers=settings.WORKERS,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test,
        batch_size=batch_size,
        num_workers=settings.WORKERS,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    return train_loader, val_loader, test_loader