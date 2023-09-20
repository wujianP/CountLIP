conda activate /discobox/wjpeng/env/clip
cd /discobox/wjpeng/code/202306/CountLIP/annotate_data
git pull

export CUDA_VISIBLE_DEVICES=0
python annotate_data.py \
--data_root /DDN_ROOT/wjpeng/dataset/LVIS \
--lvis_ann /DDN_ROOT/wjpeng/dataset/LVIS/annotations/lvis/lvis_v1_train.json \
--coco_caption_ann /DDN_ROOT/wjpeng/dataset/LVIS/annotations/coco/captions_train2017.json \
--coco_instance_ann /DDN_ROOT/wjpeng/dataset/LVIS/annotations/coco/instances_train2017.json \
--blip_path /discobox/wjpeng/weights/blip2 \
--return_coco_ann
