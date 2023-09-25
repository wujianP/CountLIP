--train-data
--val-data
--copy-codebase
--dataset-type

conda activate /discobox/wjpeng/env/countLIP
cd /discobox/wjpeng/code/202306/CountLIP/open_clip/src/training
git pull

CUDA_VISIBLE_DEVICES=0 python -m main \
    --data-root /DDN_ROOT/wjpeng/dataset \
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
    --warmup 50
