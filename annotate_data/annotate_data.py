"""
generate caption of an image, and then detect the objects, and then generate a caption that contains specific numbers in it
"""
import torch
import wandb
import argparse
import torchvision

import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as TS

import sys

sys.path.append('../')
from dataset import LVISDataset, lvis_collate_fn
from torch.utils.data import DataLoader
from collections import Counter
from transformers import Blip2Processor, Blip2ForConditionalGeneration

# grounded DINO
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap


def load_grounding_dino_model(model_config_path, model_checkpoint_path):
    """load groundingdino model"""
    cfg = SLConfig.fromfile(model_config_path)
    cfg.device = "cuda"
    model = build_model(cfg)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model.cuda()


def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True):
    # preprocess captions
    for k in range(len(caption)):
        caption[k] = caption[k].lower().strip()
        if not caption[k].endswith("."):
            caption[k] = caption[k] + "."
    # forward grounded dino
    with torch.no_grad():
        outputs = model(image, captions=caption)
    logits = outputs["pred_logits"].cpu().sigmoid()  # (bs, nq, 256)
    boxes = outputs["pred_boxes"].cpu()  # (bs, nq, 4)
    # post process
    boxes_list, scores_list, phrases_list = [], [], []
    for ub_logits, ub_boxex, cap in zip(logits, boxes, caption):
        mask = ub_logits.max(dim=1)[0] > box_threshold
        logits_filtered = ub_logits[mask]  # (n, 256)
        boxes_filtered = ub_boxex[mask]  # (n, 4)
        phrases_filtered = []
        scores_filtered = []
        for logit, box in zip(logits_filtered, boxes_filtered):
            # wj: only keep the most confident one
            posmap = (logit > text_threshold) * (logit == logit.max())
            pred_phrase = get_phrases_from_posmap(posmap, model.tokenizer(cap), model.tokenizer)
            if with_logits:
                phrases_filtered.append(pred_phrase + f"({str(logit.max().item())[:4]})")
            else:
                phrases_filtered.append(pred_phrase)
            scores_filtered.append(logit.max().item())
        boxes_list.append(boxes_filtered)
        scores_list.append(torch.Tensor(scores_filtered))
        phrases_list.append(phrases_filtered)
    return boxes_list, scores_list, phrases_list


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
                          coco_caption_ann=args.coco_caption_ann,
                          coco_instance_ann=args.coco_instance_ann,
                          return_coco_ann=args.return_coco_ann)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            pin_memory=True,
                            shuffle=True,
                            collate_fn=lvis_collate_fn)

    blip_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b",
                                                    cache_dir=args.blip_path)
    blip_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b",
                                                               torch_dtype=torch.float16,
                                                               cache_dir=args.blip_path,
                                                               device_map="auto")

    ground_dino_model = load_grounding_dino_model(args.grounded_dino_config, args.grounded_dino_path)

    for cur_idx, (img_list, boxes_list, masks_list, areas_list, cats_list, captions_list) in enumerate(dataloader):

        # BLIP2 caption
        blip_inputs = blip_processor(img_list, return_tensors="pt").to("cuda", torch.float16)
        blip_out = blip_model.generate(**blip_inputs)
        blip_captions = blip_processor.batch_decode(blip_out, skip_special_tokens=True)
        torch.cuda.empty_cache()
        print(blip_captions)

        # Grounded-DINO detection
        trans_grounded = TS.Compose(
            [
                TS.Resize((800, 800)),
                TS.ToTensor(),
                TS.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        dino_images = torch.stack([trans_grounded(img) for img in img_list], dim=0).cuda()

        from IPython import embed
        embed()

        # > forward grounded dino >
        boxes_filt_list, scores_list, pred_phrases_list = get_grounding_output(
            model=ground_dino_model,
            image=dino_images,
            caption=blip_captions,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold)
        torch.cuda.empty_cache()

        # > post process bounding box >
        for i in range(len(boxes_filt_list)):
            H, W = Hs[i], Ws[i]
            boxes = boxes_filt_list[i]
            for k in range(boxes.size(0)):
                boxes[k] = boxes[k] * torch.Tensor([W, H, W, H])
                boxes[k][:2] -= boxes[k][2:] / 2
                boxes[k][2:] += boxes[k][:2]
            boxes_filt_list[i] = boxes.cuda()
        # > use NMS to handle overlapped boxes >
        for i in range(args.batch_size):
            boxes_filt_list[i] = boxes_filt_list[i].cpu()
            nms_idx = torchvision.ops.nms(boxes_filt_list[i], scores_list[i], args.iou_threshold).numpy().tolist()
            boxes_filt_list[i] = boxes_filt_list[i][nms_idx].cuda()
            pred_phrases_list[i] = [pred_phrases_list[i][idx] for idx in nms_idx]
            # caption = check_caption(tag2text_caption, pred_phrases)
        # empty cache
        torch.cuda.empty_cache()

        # analysis categories
        object_count_list = [dict(Counter(cats)) for cats in cats_list]
        # visualize
        for i in range(args.batch_size):
            wandb_visualize(img_list[i], boxes_list[i], masks_list[i], areas_list[i],
                            object_count_list[i], cats_list[i], captions_list[i])


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Annotate Data')
    parser.add_argument('--data_root', type=str)
    parser.add_argument('--blip_path', type=str)
    parser.add_argument('--grounded_dino_config', type=str)
    parser.add_argument('--grounded_dino_path', type=str)
    parser.add_argument('--lvis_ann', type=str)
    parser.add_argument('--coco_caption_ann', type=str)
    parser.add_argument('--coco_instance_ann', type=str)
    parser.add_argument('--return_coco_ann', action='store_true')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=8)
    args = parser.parse_args()
    wandb.login(key='8cff0498531e0409db5f3c43b52a26b0d068f2dc')
    run = wandb.init('Annotate COCO Dataset')
    main()
    wandb.finish()

# img2dataset --url_list metadata/00 --input_format "parquet"\
#          --url_col "URL" --caption_col "TEXT" --output_format webdataset\
#            --output_folder laion400m-data --processes_count 20 --thread_count 128 --image_size 256\
#              --save_additional_columns '["NSFW","similarity","LICENSE"]' --enable_wandb False
