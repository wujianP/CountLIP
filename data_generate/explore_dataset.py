"""
Visualize and Explore Existing Detection and Counting Dataset to Have an Overall Understanding of Them
1. LVIS
2.
"""
import torch
import wandb
import argparse

import matplotlib.pyplot as plt
import numpy as np

from dataset import LVISDataset, lvis_collate_fn
from torch.utils.data import DataLoader
from collections import Counter


def show_box(box, ax, label):
    color = np.concatenate([np.random.random(3), np.array([0.8])], axis=0)
    x0, y0, w, h = box[0], box[1], box[2], box[3]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=color, facecolor=(0, 0, 0, 0), lw=3))
    ax.text(x0, y0, label)


def wandb_visualize(img, boxes, masks, areas, object_count, cats, caps):
    w, h = img.size
    # img-boxs
    plt.figure(figsize=(w / 60, h / 60))
    ax1 = plt.gca()
    ax1.axis('off')
    ax1.imshow(img)
    if len(boxes) > 0:
        for (box, label) in zip(boxes, cats):
            show_box(box, ax1, label)
    fig_img_box = plt.gcf()

    obj_cnt_str = "\n".join(f"{key}: {value}" for key, value in object_count.items())
    cap_str = "\n\n".join(caps)

    plt.close()

    run.log({'Image': wandb.Image(fig_img_box, caption=obj_cnt_str + '\n' + cap_str)})


@torch.no_grad()
def main():
    dataset = LVISDataset(data_root=args.data_root,
                          lvis_ann=args.lvis_ann,
                          coco_ann=args.coco_ann,
                          return_coco_ann=True)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            pin_memory=True,
                            shuffle=True,
                            collate_fn=lvis_collate_fn)

    for cur_idx, (img_list, boxes_list, masks_list, areas_list, cats_list, captions_list) in enumerate(dataloader):
        # analysis categories
        object_count_list = [dict(Counter(cats)) for cats in cats_list]

        # visualize
        for i in range(args.batch_size):
            wandb_visualize(img_list[i], boxes_list[i], masks_list[i], areas_list[i],
                            object_count_list[i], cats_list[i], captions_list[i])


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Explore LVIS Dataset')
    parser.add_argument('--data_root', type=str)
    parser.add_argument('--lvis_ann', type=str)
    parser.add_argument('--coco_ann', type=str)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=8)
    args = parser.parse_args()

    wandb.login(key='8cff0498531e0409db5f3c43b52a26b0d068f2dc')
    run = wandb.init('Explore LVIS Dataset')
    main()
    wandb.finish()
