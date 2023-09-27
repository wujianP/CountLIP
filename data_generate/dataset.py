import os
import random

import xml.etree.ElementTree as ET

from PIL import Image
from torch.utils.data import Dataset


class ImageNet(Dataset):
    def __init__(self, data_root):
        """
        :param data_root: the root path of the dataset, default: /dev/shm/imagenet/train
        """
        self.data_root = data_root

        self.image_path_list = []
        for class_id in os.listdir(os.path.join(data_root)):
            for img_name in os.listdir(os.path.join(data_root, class_id)):
                self.image_path_list.append(os.path.join(self.data_root, class_id, img_name))

    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self, idx):
        image_path = self.image_path_list[idx]
        image = Image.open(image_path).convert('RGB')

        return image


class ImageNetWithBox(Dataset):
    def __init__(self, data_root):
        """
        This is designed for ImageNet-Boxes
        :param data_root: the root path of the dataset, default: /dev/shm/imagenet
        """
        self.data_root = data_root
        # imagenet class id to class name, eg: 'n01440764' -> ['tench', 'Tinca tinca']
        self.id2class = {}
        with open(os.path.join(data_root, 'id2class.txt'), 'r') as file:
            for line in file.readlines():
                classId = line.strip()[:9]
                className = line.strip()[10:].split(', ')
                self.id2class[classId] = className

        # imagenet file list, ['n02791124_6215.JPEG', ..., 'n02791124_9967.JPEG', ...]
        self.image_name_list = []
        for class_id in os.listdir(os.path.join(data_root, 'boxes')):
            for img_name in os.listdir(os.path.join(data_root, 'boxes', class_id)):
                self.image_name_list.append(img_name.split('.')[0])

    def __len__(self):
        return len(self.image_name_list)

    def __getitem__(self, idx):
        image_name = self.image_name_list[idx]
        class_id = image_name.split('_')[0]

        class_name = random.choice(self.id2class[class_id])

        image_path = os.path.join(self.data_root, 'train', class_id, image_name) + '.JPEG'
        image = Image.open(image_path).convert('RGB')

        box_path = os.path.join(self.data_root, 'boxes', class_id, image_name) + '.xml'
        box_ann = ET.parse(box_path).getroot()
        x_min = int(box_ann.find('object').find('bndbox').find('xmin').text)
        y_min = int(box_ann.find('object').find('bndbox').find('ymin').text)
        x_max = int(box_ann.find('object').find('bndbox').find('xmax').text)
        y_max = int(box_ann.find('object').find('bndbox').find('ymax').text)

        box = [x_min, y_min, x_max, y_max]

        return image, box, class_name, class_id
