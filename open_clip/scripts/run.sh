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
    --epochs=5000 \
    --lr=5e-7 \
    --pretrained="openai" \
    --model="ViT-B-32"\
    --workers 8 \
    --copy-codebase \
    --warmup 50

conda activate /discobox/wjpeng/env/countLIP
cd /discobox/wjpeng/code/202306/CountLIP/open_clip/src/training
git pull

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
torchrun --nproc_per_node 8 -m main \
    --logs="/DDN_ROOT/wjpeng/ckp/CountLIP/imagenet_only" \
    --name="inter_hn-2_fill-mean_bs256*8_ep5_lr1e-6_warm100_vit-b-32" \
    --dataset-type="count" \
    --count-loss-type="inter" \
    --count-loss-weight=1. \
    --hard-num=2 \
    --empty-fill-type="mean" \
    --batch-size=256 \
    --epochs=5 \
    --lr=1e-6 \
    --warmup 100 \
    --pretrained="openai" \
    --model="ViT-B-32"\
    --workers 6 \
    --copy-codebase \
    --data-root /dev/shm/imagenet \
    --train-data="zhan-wei-fu" \
    --log-every-n-steps 10
