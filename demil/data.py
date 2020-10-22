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
            captions_tensors = self.tokenizer.batch_encode_plus(
                captions,
                add_special_tokens=True,
                max_length=100,
                padding="max_length",
                return_tensors="pt",
                return_attention_mask=True,
                truncation=True,
            )
        images = torch.stack(images)
        return (captions_tensors, images, score)


# data is a list containing the batch examples, where each element is an example. data[i] is a list with size 3, where index 0 is the textual data, index 1 is visual data, and index 2 is the label
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
        posts_mask = torch.zeros(settings.MAX_SEQ_LENGTH, dtype=torch.bool, device=attn_mask.device)
        posts_mask[:fill] = True
        n_tokens = input_ids.size(1)
        input_ids_pad = []
        attn_mask_pad = []

        vis_padding = torch.zeros(size=[fill, *visual_data.size()[1:]])
        batch_visual_data.append(torch.cat([vis_padding, visual_data], 0))
        batch_post_attn_mask.append(posts_mask)

        for _ in range(0, fill):
            # TODO: This works for RoBERTa, but I don't know if it works for BERTimbau
            # This is the representation of an empty sentence in BERT.
            # All ones, except the first and second elements being 0 and 2 respectively
            # TODO:
            aux_input_ids = torch.ones(
                size=(1, n_tokens), device=input_ids.device, dtype=input_ids.dtype
            )
            aux_input_ids[0][0], aux_input_ids[0][1] = 0, 2
            input_ids_pad.append(aux_input_ids)
            # This is the representation of an empty sentence for the
            # attention mask in BERT: all zeros except indexes 0 and 1 being 1.
            aux_attn_mask = torch.zeros(
                size=(1, n_tokens), device=input_ids.device, dtype=input_ids.dtype
            )
            aux_attn_mask[0][0], aux_attn_mask[0][1] = 1, 1
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

