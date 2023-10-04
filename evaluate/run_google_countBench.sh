conda activate /discobox/wjpeng/env/countLIP
cd /discobox/wjpeng/code/202306/CountLIP/evaluate
git pull

python main_google_benchmark.py \
--data-root='/DDN_ROOT/wjpeng/dataset/countBench/google/data' \
--batch-size=16 \
--num-workers=8 \
--model-name="ViT-B-32" \
--resume='/workspace/b32_400m.pt'
--resume='/DDN_ROOT/wjpeng/ckp/CountLIP/imagenet_only/inter_hn-2_fill-black_bs256*8_ep10_lr5e-6_warm100_vit-b-32-openai/checkpoints/epoch_5.pt'

