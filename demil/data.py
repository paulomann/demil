import torch
from typing import Literal, Tuple
from transformers import PreTrainedTokenizer
from torchvision import transforms
from PIL import Image
import demil.utils as util
from demil import settings


class DepressionCorpus(torch.utils.data.Dataset):
    def __init__(
        self,
        period: Literal[60, 212, 365],
        dataset_type: Literal["train", "val", "test"],
        tokenizer: PreTrainedTokenizer,
        regression: bool = False,
        max_seq_length: int = settings.MAX_SEQ_LENGTH,
    ):
        self.dataset = util.get_dataset(period, dataset_type)
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
    for i, example in enumerate(data):

        textual_data, visual_data, label = example
        input_ids, attn_mask = textual_data["input_ids"], textual_data["attention_mask"]
        fill = settings.MAX_SEQ_LENGTH - input_ids.size(0)
        posts_mask = torch.zeros(
            settings.MAX_SEQ_LENGTH, dtype=torch.bool, device=attn_mask.device
        )
        posts_mask[:fill] = True
        n_tokens = input_ids.size(1)
        input_ids_pad = []
        attn_mask_pad = []

        vis_padding = torch.zeros(size=[fill, *visual_data.size()[1:]])
        batch_visual_data.append(torch.cat([vis_padding, visual_data], 0))
        batch_post_attn_mask.append(posts_mask)

        for _ in range(0, fill):
            aux_input_ids, aux_attn_mask = get_input_ids_attn_mask(
                n_tokens,
                device=input_ids.device,
                i_ids_dtype=input_ids.dtype,
                attn_dtype=attn_mask.dtype,
            )
            input_ids_pad.append(aux_input_ids)
            attn_mask_pad.append(aux_attn_mask)

        input_ids_pad.append(input_ids)
        attn_mask_pad.append(attn_mask)
        batch_input_ids.append(torch.cat(input_ids_pad, 0))
        batch_attn_mask_ids.append(torch.cat(attn_mask_pad, 0))

        labels[i] = label

    return (
        (torch.stack(batch_input_ids), torch.stack(batch_attn_mask_ids)),
        torch.stack(batch_visual_data),
        torch.stack(batch_post_attn_mask),
        labels,
    )
