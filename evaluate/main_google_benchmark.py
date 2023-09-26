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

    correct_num = 0
    total_dist = 0
    for cur_idx, (images, all_texts, labels) in enumerate(dataloader):
        B, C, L = all_texts.shape   # B: batch size, C: 2-10, L: sentence length

        all_texts = all_texts.view(-1, L)    # [B, 9, 77] -> [9B, 77] B: batch size, 9: 2-10, 77: sentence length
        images, all_texts = images.cuda(), all_texts.cuda()

        model_out = model(images, all_texts)
        image_feats = model_out[0]  # [B, 512]  B: batch size, 512: feat dim
        text_feats = model_out[1]   # [9B, 512]
        logit_scale = model_out[2]

        # normalize feats

        text_feats = text_feats.view(B, C, -1)  # [9B, 512] -> [B, 9, 512]

        # compute similarity
        for i in range(B):
            query_image_feat = image_feats[i]  # [512]
            key_text_feats = text_feats[i]  # [9, 512]
            img2text_sim = logit_scale * query_image_feat @ key_text_feats.T

            pred_label = img2text_sim.argmax().item()
            gt_label = labels[i].item()

            dist = abs(pred_label - gt_label)
            total_dist += dist
            if pred_label == gt_label:
                correct_num += 1

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
    
