import argparse

import torch

from dataset import GoogleCountBench
from torch.utils.data import DataLoader
from open_clip import create_model_and_transforms, get_tokenizer


def my_collate_fn(batch):
    images, anns = [], []
    for sample in batch:
        images.append(sample[0])
        anns.append(sample[1])
    return images, anns


@torch.no_grad()
def main():

    # load model and checkpoint
    model, _, val_trans = create_model_and_transforms(
        model_name=args.model_name,
        pretrained=args.resume,
        device="cuda",
    )
    tokenizer = get_tokenizer(args.model_name)

    # load dataset
    dataset = GoogleCountBench(data_root=args.data_root, transform=val_trans, tokenizer=tokenizer)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False,
        drop_last=False
    )

    for cur_idx, (images, anns) in enumerate(dataloader):
        from IPython import embed
        embed()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, default='/DDN_ROOT/wjpeng/dataset/countBench/google/data')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument("--model-name", default="ViT-B-32", type=str)
    parser.add_argument("--resume", type=str, help='path to the model checkpoint')
    args = parser.parse_args()

    main()
    
