import torch
import os
import json
import inflect

from torch.utils.data import Dataset, DataLoader
from PIL import Image


def my_collate_fn(batch):
    images, anns = [], []
    for sample in batch:
        images.append(sample[0])
        anns.append(sample[1])
    return images, anns


@torch.no_grad()
def evaluate(model, val_trans, tokenizer, ):

    # load dataset
    dataset = GoogleCountBench(data_root='/DDN_ROOT/wjpeng/dataset/countBench/google/data',
                               transform=val_trans,
                               tokenizer=tokenizer)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=16,
        num_workers=8,
        pin_memory=True,
        shuffle=False,
        drop_last=False
    )

    correct_num = 0
    total_dist = 0
    for cur_idx, (images, all_texts, labels) in enumerate(dataloader):
        B, C, L = all_texts.shape   # B: batch size, C: 2-10, L: sentence length

        all_texts = all_texts.view(-1, L)    # [B, 9, 77] -> [9B, 77] B: batch size, 9: 2-10, 77: sentence length
        images, all_texts = images.cuda(), all_texts.cuda()

        model_out = model(images, all_texts)
        image_feats = model_out[0]  # [B, 512]  B: batch size, 512: feat dim
        text_feats = model_out[1]   # [9B, 512]
        logit_scale = model_out[2]

        # normalize feats

        text_feats = text_feats.view(B, C, -1)  # [9B, 512] -> [B, 9, 512]

        # compute similarity
        for i in range(B):
            query_image_feat = image_feats[i]  # [512]
            key_text_feats = text_feats[i]  # [9, 512]
            img2text_sim = logit_scale * query_image_feat @ key_text_feats.T

            pred_label = img2text_sim.argmax().item()
            gt_label = labels[i].item()

            dist = abs(pred_label - gt_label)
            total_dist += dist
            if pred_label == gt_label:
                correct_num += 1
    acc = correct_num / len(dataset) * 100
    dist = total_dist / len(dataset)

    return acc, dist


class GoogleCountBench(Dataset):
    def __init__(self, data_root, transform, tokenizer):
        self.data_root = data_root
        self.transform = transform
        self.tokenize = tokenizer

        files = os.listdir(self.data_root)
        self.image_list = []
        self.ann_list = []
        for file in files:
            if file.endswith('.jpg'):
                self.image_list.append(file)
            if file.endswith('.json'):
                self.ann_list.append(file)
        self.image_list = sorted(self.image_list)
        self.ann_list = sorted(self.ann_list)

        # check correctness
        assert len(self.ann_list) == len(self.image_list)
        for img, ann in zip(self.image_list, self.ann_list):
            assert img.split('.')[0] == ann.split('.')[0]

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        image_path = os.path.join(self.data_root, image_name)
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        ann_name = self.ann_list[idx]
        ann_path = os.path.join(self.data_root, ann_name)
        with open(ann_path, 'r') as file:
            ann = json.load(file)
        file.close()

        all_texts = self.generate_all_text(ann)
        all_texts = self.tokenize(all_texts)

        label = ann['number'] - 2   # convert number to the index in all_texts

        return image, all_texts, label

    @staticmethod
    def generate_all_text(ann):
        origin_text = ann['text'].lower()
        number_word = ann['number_word'].lower()
        p = inflect.engine()
        all_number_words = [p.number_to_words(num) for num in range(2, 11)]
        all_texts = [origin_text.replace(number_word, new_num_word) for new_num_word in all_number_words]

        return all_texts
