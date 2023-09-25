#--val-data

conda activate /discobox/wjpeng/env/countLIP
cd /discobox/wjpeng/code/202306/CountLIP/open_clip/src/training
git pull
rm -rf /DDN_ROOT/wjpeng/ckp/CountLIP/debug
CUDA_VISIBLE_DEVICES=0 python -m main \
    --dataset-type="count" \
    --count-loss-type="intra" \
    --count-loss-weight=1. \
    --data-root /dev/shm/imagenet \
    --train-data="zhan-wei-fu" \
    --hard-num=2 \
    --empty-fill-type="mean" \
    --logs="/DDN_ROOT/wjpeng/ckp/CountLIP/debug" \
    --name="countLIP-debug" \
    --batch-size=128 \
    --epochs=5 \
    --lr=5e-7 \
    --pretrained="openai" \
    --model="ViT-B-32"\
    --workers 8 \
    --copy-codebase \
    --warmup 50
