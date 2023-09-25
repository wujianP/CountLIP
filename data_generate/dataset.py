import os

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
