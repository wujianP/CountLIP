JOB_ID=0
conda activate /discobox/wjpeng/env/countLIP
cd /discobox/wjpeng/code/202306/CountLIP/data_generate
git pull

export CUDA_VISIBLE_DEVICES=$((JOB_ID % 8))
python segment_imagenet.py \
--data_root='/dev/shm/imagenet/' \
--out_dir='/DDN_ROOT/wjpeng/dataset/imagenet/segment' \
--batch_size=8 \
--num_workers=8 \
--sam_checkpoint='/discobox/wjpeng/weights/sam/sam_vit_l_0b3195.pth' \
--job_num=24 \
--job_id=$JOB_ID