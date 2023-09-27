from dataset import ImageNetWithBox
from torch.utils.data import DataLoader

import torch
import argparse

# segment anything
from segment_anything import build_sam, SamPredictor
from segment_anything.utils.transforms import ResizeLongestSide


def my_collate_fn(batch):
    images, boxs, class_names, class_ids = [], [], [], []
    for sample in batch:
        images.append(sample[0])
        boxs.append(sample[1])
        class_names.append(sample[2])
        class_ids.append(sample[3])
    return images, boxs, class_names, class_ids


@torch.no_grad()
def main():

    # load sam
    sam = build_sam(checkpoint=args.sam_checkpoint).cuda()

    # load dataset
    dataset = ImageNetWithBox(data_root=args.data_root)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=my_collate_fn,
        pin_memory=True,
        drop_last=False,
        shuffle=False
    )

    total_iter = len(dataloader)
    for cur_iter, (images, boxs, class_names, class_ids) in enumerate(dataloader):
        from IPython import embed
        embed()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('SAM segment ImageNet')
    parser.add_argument('--data_root', type=str, default='/dev/shm/imagenet/')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--sam_checkpoint', type=str, default='/discobox/wjpeng/weights/sam/sam_vit_h_4b8939.pth')
    args = parser.parse_args()
    device = 'cuda'
    main()
