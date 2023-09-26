conda activate /discobox/wjpeng/env/countLIP
cd /discobox/wjpeng/code/202306/CountLIP/evaluate
git pull

python main_google_benchmark.py \
--data-root='/DDN_ROOT/wjpeng/dataset/countBench/google/data' \
--batch-size=16 \
--num-workers=8 \
--model-name="ViT-B-32" \
--resume='/DDN_ROOT/wjpeng/ckp/CountLIP/imagenet_only/inter_hn-2_fill-mean_bs256*8_ep5_lr1e-6_warm100_vit-b-32/checkpoints/epoch_1.pt'
