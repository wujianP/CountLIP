import wandb
import os
import random
import json

from PIL import Image


def explore_fsc147(enable_wandb=False):
    
    img_filenames = os.listdir(img_root)
    random.shuffle(img_filenames)

    img2cls_path = os.path.join(ann_root, 'ImageClasses_FSC147.txt')
    img2cls = {}
    with open(img2cls_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            img, cls = line.strip().split('\t')
            img2cls[img] = cls
    f.close()

    ann_path = os.path.join(ann_root, 'annotation_FSC147_384.json')
    with open(ann_path, 'r') as f:
        anns = json.load(f)
    f.close()

    metadata = {}
    for img_filename in img_filenames:
        path = os.path.join(img_root, img_filename)
        img = Image.open(path).convert('RGB')
        cls = img2cls[img_filename]
        ann = anns[img_filename]
        points = ann['points']
        count = len(points)

        metadata[path] = {
            'class': cls,
            'count': count
        }

        if enable_wandb:
            run.log({'FSC-147': wandb.Image(img, caption=f'class: {cls}  -- count: {count}')})

    return metadata


def volume_up_fsc147(info):
    cnt2imgs = {}
    for img, ann in info.items():
        cnt = ann['count']
        if cnt in cnt2imgs.keys():
            cnt2imgs[cnt].append(img)
        else:
            cnt2imgs[cnt] = [img]
    return cnt2imgs


if __name__ == '__main__':
    img_root = '/DDN_ROOT/wjpeng/dataset/FSC-147/images_384_VarV2'
    ann_root = '/DDN_ROOT/wjpeng/dataset/FSC-147/annotations'

    wandb.login()
    run = wandb.init('FSC-147')

    datainfo = explore_fsc147()

    countinfo = volume_up_fsc147(datainfo)

    from IPython import embed
    embed()
