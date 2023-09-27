import os

from dataset import ImageNetWithBox
from torch.utils.data import DataLoader, Subset
from PIL import Image, ImageDraw

import torch
import open_clip
import argparse
import wandb
import time
import numpy as np
import matplotlib.pyplot as plt

# segment anything
from segment_anything import build_sam, SamPredictor, build_sam_vit_l
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


def wandb_visualize(images, class_names, similarities, boxes, foregrounds):
    for img, cls, sim, box, fg in zip(images, class_names, similarities, boxes, foregrounds):
        draw = ImageDraw.Draw(img)
        coords = tuple(box)
        draw.rectangle(coords, outline="red", width=2)

        run.log({'segment': [wandb.Image(img, caption=cls), wandb.Image(fg, caption=f'similarity: {sim:2f}')]})


@torch.no_grad()
def main():

    # load sam
    sam = build_sam_vit_l(checkpoint=args.sam_checkpoint).cuda()

    # load clip
    # clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(args.clip_model,
    #                                                                        pretrained='laion2b_s34b_b88k')
    # clip_model.cuda()
    # clip_tokenizer = open_clip.get_tokenizer(args.clip_model)

    # load dataset
    dataset = ImageNetWithBox(data_root=args.data_root)

    total_len = len(dataset)
    split_len = total_len // args.job_num
    start_idx = args.job_id * split_len
    end_idx = start_idx + split_len
    if args.job_id == (args.job_num - 1):
        end_idx = total_len
    split_dataset = Subset(dataset, range(start_idx, end_idx))

    print(f'dataset split: start index: {start_idx}, end index: {end_idx}, split len: {len(split_dataset)}')

    dataloader = DataLoader(
        dataset=split_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=my_collate_fn,
        pin_memory=True,
        drop_last=False,
        shuffle=True
    )

    total_iter = len(dataloader)
    start_time = time.time()
    for cur_iter, (images, boxs, class_names, class_ids, filenames) in enumerate(dataloader):
        data_time = time.time() - start_time
        start_time = time.time()

        # sam segment
        batched_input = prepare_sam_data(images=images, boxes=boxs, resize_size=sam.image_encoder.img_size)
        batched_output = sam(batched_input, multimask_output=False)
        masks_list = [output['masks'][0].cpu().numpy() for output in batched_output]
        torch.cuda.empty_cache()

        # extract foreground
        foreground_pils = []
        for img, mask, box in zip(images, masks_list, boxs):
            mask_pil = Image.fromarray((mask[0] * 255).astype(np.uint8), mode='L')
            bg = Image.new('RGB', img.size, (0, 0, 0))
            fg = Image.composite(img, bg, mask_pil)
            fg = fg.crop(box)
            foreground_pils.append(fg)

        sam_time = time.time() - start_time
        start_time = time.time()

        # calculate text-foreground similarity

        # clip_images = [clip_preprocess(img) for img in images]
        # clip_images = torch.stack(clip_images, dim=0).cuda()
        # clip_texts = clip_tokenizer([f'a {cls}' for cls in class_names]).cuda()
        #
        # with torch.no_grad():
        #     image_features = clip_model.encode_image(clip_images)
        #     text_features = clip_model.encode_text(clip_texts)
        #     image_features /= image_features.norm(dim=-1, keepdim=True)
        #     text_features /= text_features.norm(dim=-1, keepdim=True)
        #     similarity = (image_features * text_features).sum(dim=1)
        # torch.cuda.empty_cache()

        # clip_time = time.time() - start_time
        # start_time = time.time()

        # wandb_visualize(images, class_names, similarity, boxs, foreground_pils)
        # vis_time = time.time() - start_time
        # start_time = time.time()

        # save
        for i in range(len(images)):
            fg_pil = foreground_pils[i]
            cls_id = class_ids[i]
            fn = filenames[i]
            dir_path = os.path.join(args.out_dir, cls_id)
            os.makedirs(dir_path, exist_ok=True)
            fg_pil.save(os.path.join(dir_path, fn) + '.jpg')
        save_time = time.time() - start_time
        start_time = time.time()

        print(f'[JOB: {args.job_id} / {args.job_num}  {cur_iter+1} / {total_iter}] ({(cur_iter+1)/total_iter*100:.2f}%) '
              f'data: {data_time:.2f} sam: {sam_time:.2f} save: {save_time:.2f} '
              f'batch: {data_time+sam_time+save_time:.2f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('SAM segment ImageNet')
    parser.add_argument('--data_root', type=str, default='/dev/shm/imagenet/')
    parser.add_argument('--out_dir', type=str, default='/DDN_ROOT/wjpeng/dataset/imagenet/segment')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--sam_checkpoint', type=str, default='/discobox/wjpeng/weights/sam/sam_vit_l_0b3195.pth')
    parser.add_argument('--clip_model', type=str, default='ViT-B-16')

    parser.add_argument('--job_num', type=int, default=1)
    parser.add_argument('--job_id', type=int, default=0)
    args = parser.parse_args()
    device = 'cuda'

    # wandb.login()
    # run = wandb.init('SAM ImageNet')
    main()
    # run.finish()
