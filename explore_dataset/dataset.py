import os

from PIL import Image
from torch.utils.data import Dataset
from lvis import LVIS
from pycocotools.coco import COCO


class LVISDataset(Dataset):
    def __init__(self, data_root, lvis_ann, coco_caption_ann, coco_instance_ann, return_coco_ann):
        self.data_root = data_root
        self.lvis = LVIS(lvis_ann)
        self.coco_caption = COCO(coco_caption_ann)
        self.coco_instance = COCO(coco_instance_ann)
        self.image_ids = self.lvis.get_img_ids()
        self.return_coco_ann = return_coco_ann

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        # load image
        img_id = self.image_ids[idx]
        img_dict = self.lvis.load_imgs([img_id])[0]
        img_filename = '/'.join(img_dict['coco_url'].split('/')[-2:])
        img_path = os.path.join(self.data_root, img_filename)
        img = Image.open(img_path).convert('RGB')

        # load caption
        captions = [cap['caption'] for cap in self.coco_caption.imgToAnns[img_id]]

        # load mask, box... using annotation from COCO >
        if self.return_coco_ann:
            ann_dicts = self.coco_instance.imgToAnns[img_id]
            boxes, masks, areas, cats = [], [], [], []
            for ann in ann_dicts:
                mask = self.coco_instance.annToMask(ann)
                box = ann['bbox']
                area = ann['area']
                cat_id = ann['category_id']
                cat = self.coco_instance.loadCats(cat_id)[0]['name']
                boxes.append(box)
                masks.append(mask)
                areas.append(area)
                cats.append(cat)
        # load mask, box... using annotation from LVIS >
        else:
            ann_ids = self.lvis.get_ann_ids(img_ids=[img_id])
            ann_dicts = self.lvis.load_anns(ann_ids)
            boxes, masks, areas, cats = [], [], [], []
            for ann_dict in ann_dicts:
                mask = self.lvis.ann_to_mask(ann_dict)
                box = ann_dict['bbox']  # [x,y,w,h]
                area = ann_dict['area']
                cat_id = ann_dict['category_id']
                cat = self.lvis.load_cats([cat_id])[0]['name']
                boxes.append(box)
                masks.append(mask)
                areas.append(area)
                cats.append(cat)

        return img, boxes, masks, areas, cats, captions


def lvis_collate_fn(batch):
    img_list, boxes_list, masks_list, areas_list, cats_list, captions_list = [], [], [], [], [], []
    for item in batch:
        img_list.append(item[0])
        boxes_list.append(item[1])
        masks_list.append(item[2])
        areas_list.append(item[3])
        cats_list.append(item[4])
        captions_list.append(item[5])
    return img_list, boxes_list, masks_list, areas_list, cats_list, captions_list
