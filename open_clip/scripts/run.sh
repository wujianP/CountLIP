#--val-data
#
conda activate /discobox/wjpeng/env/countLIP
cd /discobox/wjpeng/code/202306/CountLIP/open_clip/src/training
git pull
rm -rf /DDN_ROOT/wjpeng/ckp/CountLIP/debug
CUDA_VISIBLE_DEVICES=0 python -m main \
    --dataset-type="mix" \
    --train-data="/DDN_ROOT/wjpeng/dataset/CC3M/data/{00000..00331}.tar" \
    --train-num-samples=100000 \
    --count-loss-type="intra" \
    --count-loss-weight=1. \
    --data-root /dev/shm/imagenet \
    --hard-num=2 \
    --empty-fill-type="black" \
    --logs="/DDN_ROOT/wjpeng/ckp/CountLIP/debug" \
    --name="countLIP-debug" \
    --batch-size=128 \
    --epochs=5000 \
    --lr=5e-7 \
    --pretrained="openai" \
    --model="ViT-B-32"\
    --model-root="/discobox/wjpeng/weights/clip" \
    --workers 8 \
    --copy-codebase \
    --warmup 50



conda activate /discobox/wjpeng/env/countLIP
cd /discobox/wjpeng/code/202306/CountLIP/open_clip/src/training
git pull

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
torchrun --nproc_per_node 8 -m main \
    --segmented-object \
    --logs="/DDN_ROOT/wjpeng/ckp/CountLIP/waiting" \
    --name="seg-real-bg-1e-6_" \
    --dataset-type="mix" \
    --count-loss-type="inter" \
    --count-loss-weight=1. \
    --hard-num=4 \
    --empty-fill-type="real" \
    --batch-size=128 \
    --epochs=2 \
    --lr=1e-6 \
    --warmup 100 \
    --pretrained="openai" \
    --model="ViT-B-32"\
    --model-root="/discobox/wjpeng/weights/clip" \
    --workers 8 \
    --copy-codebase \
    --count-data-root="/dev/shm/imagenet" \
    --count-background-root="/DDN_ROOT/wjpeng/dataset/BG-20k/train" \
    --train-data="zhan-wei-fu" \
    --log-every-n-steps 10 \
    --eval-google-every-n-steps 20


conda activate /discobox/wjpeng/env/countLIP
cd /discobox/wjpeng/code/202306/CountLIP/open_clip/src/training
git pull

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
torchrun --nproc_per_node 8 -m main \
    --logs="/DDN_ROOT/wjpeng/ckp/CountLIP/imagenet_only" \
    --name="inter_hn-4_fill-black_bs128*8_ep5_lr1e-6_warm100_vit-b-32-openai" \
    --dataset-type="count" \
    --count-loss-type="inter" \
    --count-loss-weight=1. \
    --hard-num=4 \
    --empty-fill-type="black" \
    --batch-size=128 \
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
