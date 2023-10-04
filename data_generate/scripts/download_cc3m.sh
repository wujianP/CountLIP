"""https://github.com/rom1504/img2dataset/blob/main/dataset_examples/cc3m.md"""

conda activate /discobox/wjpeng/env/clip/
cd /DDN_ROOT/wjpeng/dataset/CC3M/

img2dataset --url_list Train_GCC-training.tsv --input_format "tsv"\
         --url_col "url" --caption_col "caption" --output_format webdataset\
           --output_folder data --processes_count 16 --thread_count 256 --resize_mode no  --incremental