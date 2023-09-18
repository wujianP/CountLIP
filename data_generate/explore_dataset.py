"""
Visualize and Explore Existing Detection and Counting Dataset to Have an Overall Understanding of Them
1. LVIS
2.
"""
import torch
import wandb
import argparse

from dataset import LVISDataset


@torch.no_grad()
def main():
    dataset = LVISDataset(data_root=args.data_root,
                          lvis_ann=args.lvis_ann,
                          coco_ann=args.coco_ann)


    from IPython import embed

    embed()

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
