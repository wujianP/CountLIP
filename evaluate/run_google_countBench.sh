conda activate /discobox/wjpeng/env/countLIP
cd /discobox/wjpeng/code/202306/CountLIP/evaluate

python main_google_benchmark.py \
--data-root='/DDN_ROOT/wjpeng/dataset/countBench/google/data' \
--batch-size=16 \
--num-workers=8 \
--model-name="ViT-B-32" \
--resume='openai'
