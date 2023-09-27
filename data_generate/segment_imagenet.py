from dataset import ImageNetWithBox
from torch.utils.data import DataLoader

import torch
import argparse
import wandb
import numpy as np
import matplotlib.pyplot as plt

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
        box = torch.tensor([boxes[i]]).cuda()
        data = {
            'image': prepare_image(images[i], resize_transform),
            'boxes': resize_transform.apply_boxes_torch(box, (h, w)),
            'original_size': (h, w)
        }
        batched_input.append(data)
    return batched_input


def show_mask(mask, ax, random_color=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, label, random_color=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=color, facecolor=(0, 0, 0, 0), lw=2))
    ax.text(x0, y0, label)


def wandb_visualize(images, class_names, boxes, masks):
    for img, cls, box, mask in zip(images, class_names, boxes, masks):
        w, h = img.size
        # box, img, mask
        plt.figure(figsize=(w / 80, h / 80))
        ax1 = plt.gca()
        ax1.axis('off')
        ax1.imshow(img)
        show_box(box, ax1, cls)
        show_mask(mask[0], ax1)
        fig1 = plt.gcf()
        plt.close()

        # mask only
        plt.figure(figsize=(w / 80, h / 80))
        ax2 = plt.gca()
        ax2.axis('off')
        show_mask(mask[0], ax2)
        fig2 = plt.gcf()

        run.log({'segment': [wandb.Image(fig1), wandb.Image(fig2)]})


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
        shuffle=True
    )

    total_iter = len(dataloader)
    for cur_iter, (images, boxs, class_names, class_ids, filenames) in enumerate(dataloader):
        # prepare sam input
        batched_input = prepare_sam_data(images=images, boxes=boxs, resize_size=sam.image_encoder.img_size)
        batched_output = sam(batched_input, multimask_output=False)
        masks_list = [output['masks'].cpu().numpy() for output in batched_output]

        wandb_visualize(images, class_names, boxs, masks_list)



if __name__ == '__main__':
    parser = argparse.ArgumentParser('SAM segment ImageNet')
    parser.add_argument('--data_root', type=str, default='/dev/shm/imagenet/')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--sam_checkpoint', type=str, default='/discobox/wjpeng/weights/sam/sam_vit_h_4b8939.pth')
    args = parser.parse_args()
    device = 'cuda'

    wandb.login()
    run = wandb.init('SAM ImageNet')
    main()
    run.finish()
