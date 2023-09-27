from dataset import ImageNetWithBox
from torch.utils.data import DataLoader

import torch
import argparse

import numpy as np

# segment anything
from segment_anything import build_sam, SamPredictor
from segment_anything.utils.transforms import ResizeLongestSide


def my_collate_fn(batch):
    images, boxs, class_names, class_ids, filenames = [], [], [], [], []
    for sample in batch:
        images.append(sample[0])
        boxs.append(sample[1])
        class_names.append(sample[2])
        class_ids.append(sample[3])
        filenames.append(sample[4])
    return images, boxs, class_names, class_ids, filenames


def prepare_sam_data(images, boxes, resize_size):
    resize_transform = ResizeLongestSide(resize_size)

    def prepare_image(image, transform):
        image = np.array(image)
        image = transform.apply_image(image)
        image = torch.as_tensor(image).cuda()
        return image.permute(2, 0, 1).contiguous()

    batched_input = []
    for i in range(len(images)):
        w, h = images[i].size
        data = {
            'image': prepare_image(images[i], resize_transform),
            'boxes': resize_transform.apply_boxes_torch(boxes[i], h, w),
            'original_size': (h, w)
        }
        batched_input.append(data)
    return batched_input


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
    for cur_iter, (images, boxs, class_names, class_ids, filenames) in enumerate(dataloader):

        from IPython import embed
        embed()
        # prepare sam input
        batched_input = prepare_sam_data(images=images, boxes=boxs, resize_size=sam.image_encoder.img_size)
        batched_output = sam(batched_input, multimask_output=False)
        masks_list = [output['masks'].cpu().numpy() for output in batched_output]


if __name__ == '__main__':
    parser = argparse.ArgumentParser('SAM segment ImageNet')
    parser.add_argument('--data_root', type=str, default='/dev/shm/imagenet/')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--sam_checkpoint', type=str, default='/discobox/wjpeng/weights/sam/sam_vit_h_4b8939.pth')
    args = parser.parse_args()
    device = 'cuda'
    main()
