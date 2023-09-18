conda activate /discobox/wjpeng/env/clip
cd /discobox/wjpeng/code/202306/CountLIP/data_generate
git pull

export CUDA_VISIBLE_DEVICES=0
python explore_dataset.py \
--data_root /DDN_ROOT/wjpeng/dataset/LVIS \
--lvis_ann /DDN_ROOT/wjpeng/dataset/LVIS/annotations/lvis/lvis_v1_train.json \
--coco_ann /DDN_ROOT/wjpeng/dataset/LVIS/annotations/coco/captions_train2017.json \
