
conda activate /discobox/wjpeng/env/countLIP
cd /discobox/wjpeng/code/202306/CountLIP/open_clip/src/training
git pull
rm -rf /DDN_ROOT/wjpeng/ckp/CountLIP/debug
CUDA_VISIBLE_DEVICES=0 python -m main \
    --dataset-type="mix" \
    --train-data="/DDN_ROOT/wjpeng/dataset/CC3M/data/{00000..00331}.tar" \
    --count-data-root='/dev/shm/imagenet' \
    --count-background-root='/DDN_ROOT/wjpeng/dataset/BG-20k/train' \
    --hard-num=2 \
    --empty-fill-type="black" \
    --segmented-object \
    --train-num-samples=100000 \
    --count-loss-type="inter" \
    --count-loss-weight=1. \
    --logs="/DDN_ROOT/wjpeng/ckp/CountLIP/debug" \
    --name="countLIP-debug" \
    --batch-size=128 \
    --count-batch-size=8 \
    --epochs=5000 \
    --lr=1e-6 \
    --pretrained="openai" \
    --model="ViT-B-32"\
    --model-root="/discobox/wjpeng/weights/clip" \
    --workers 8 \
    --copy-codebase \
    --warmup 50