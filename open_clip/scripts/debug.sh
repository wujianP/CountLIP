conda activate /discobox/wjpeng/env/countLIP
cd /discobox/wjpeng/code/202306/CountLIP/open_clip/src/training
git pull
rm -rf /DDN_ROOT/wjpeng/ckp/CountLIP/debug
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
torchrun --nproc_per_node 8 -m main \
    --dataset-type="mix" \
    --train-data="/DDN_ROOT/wjpeng/dataset/LAION400M/laion400m-data-00-01/{00000..02587}.tar" \
    --train-num-samples=-1 \
    --dataset-resampled \
    --empty-fill-type="black" \
    --segmented-object \
    --count-loss-type="inter" \
    --count-loss-weight=0.5 \
    --batch-size=256 \
    --count-batch-size=32 \
    --hard-num=2 \
    --epochs=10 \
    --steps-per-epoch=1000 \
    --lr=1e-6 \
    --logs="/DDN_ROOT/wjpeng/ckp/CountLIP/debug" \
    --name="countLIP-debug" \
    --pretrained="openai" \
    --model="ViT-B-32"\
    --copy-codebase \
    --eval-google-every-n-steps 100 \
    --warmup 1000
