conda activate /discobox/wjpeng/env/countLIP
cd /discobox/wjpeng/code/202306/CountLIP/evaluate
git pull

python main_google_benchmark.py \
--data-root='/DDN_ROOT/wjpeng/dataset/countBench/data' \
--batch-size=64 \
--num-workers=8 \
--model-name="ViT-B-32" \
--resume='/DDN_ROOT/wjpeng/ckp/betterCLIP/new-vitb-32-openai_ep100-step100-warm2000_lr1e-6_common-bs256-cc3m-cc12m_extra-bs8-hn2-wt0.25/checkpoints/epoch_40.pt'
