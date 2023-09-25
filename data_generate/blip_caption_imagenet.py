from transformers import Blip2Processor, Blip2ForConditionalGeneration
from dataset import ImageNet
from torch.utils.data import DataLoader

import torch
import argparse


def my_collate_fn(batch):
    images = []
    for sample in batch:
        images.append(sample)
    return images


@torch.no_grad()
def main():
    # load model
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b",
                                               cache_dir=args.model_path)
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b",
        torch_dtype=torch.float32,
        cache_dir=args.model_path)
    model.cuda()

    # load dataset
    dataset = ImageNet(data_root=args.data_root)
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
    for cur_iter, images in enumerate(dataloader):
        inputs = processor(images=images, return_tensors="pt").to(device, torch.float32)
        generated_ids = model.generate(**inputs)
        output = processor.batch_decode(generated_ids, skip_special_tokens=True)
        generated_text = [text.strip() for text in output]
        from IPython import embed
        embed()

    from IPython import embed
    embed()

    # url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # image = Image.open(requests.get(url, stream=True).raw)
    #
    # inputs = processor(images=image, return_tensors="pt").to(device, torch.float32)
    #
    # generated_ids = model.generate(**inputs)
    # generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('BLIP caption ImageNet')
    parser.add_argument('--data_root', type=str, default='/dev/shm/imagenet/train')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--model_path', type=str, default='/discobox/wjpeng/weights/blip2')
    args = parser.parse_args()
    device = 'cuda'
    main()
