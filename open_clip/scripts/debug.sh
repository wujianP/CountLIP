
conda activate /discobox/wjpeng/env/countLIP
cd /discobox/wjpeng/code/202306/CountLIP/open_clip/src/training
git pull
rm -rf /DDN_ROOT/wjpeng/ckp/CountLIP/debug
CUDA_VISIBLE_DEVICES=0 python -m main \
    --dataset-type="mix" \
    --train-data="/DDN_ROOT/wjpeng/dataset/LAION400M/laion400m-data-00-01/{00000..02587}.tar" \
    --train-num-samples=100000 \
    --dataset-resampled \
    --empty-fill-type="black" \
    --segmented-object \
    --count-loss-type="inter" \
    --count-loss-weight=1. \
    --batch-size=128 \
    --count-batch-size=8 \
    --hard-num=2 \
    --epochs=5000 \
    --steps-per-epoch=500 \
    --lr=1e-6 \
    --logs="/DDN_ROOT/wjpeng/ckp/CountLIP/debug" \
    --name="countLIP-debug" \
    --pretrained="openai" \
    --model="ViT-B-32"\
    --copy-codebase \
    --warmup 5000
