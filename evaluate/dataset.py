import os
import json
import inflect

from torch.utils.data import Dataset
from PIL import Image


class GoogleCountBench(Dataset):
    def __init__(self, data_root):
        self.data_root = data_root
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

        ann_name = self.ann_list[idx]
        ann_path = os.path.join(self.data_root, ann_name)
        with open(ann_path, 'r') as file:
            ann = json.load(file)
        file.close()

        all_texts = self.generate_all_text(ann)
        ann['all_texts'] = all_texts

        return image, ann

    @staticmethod
    def generate_all_text(ann):
        origin_text = ann['text'].lower()
        number_word = ann['number_word'].lower()
        p = inflect.engine()
        all_number_words = [p.number_to_words(num) for num in range(2, 11)]
        all_texts = [origin_text.replace(number_word, new_num_word) for new_num_word in all_number_words]

        return all_texts
