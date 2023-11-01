"""https://github.com/rom1504/img2dataset/blob/main/dataset_examples/cc12m.md"""

conda activate /discobox/wjpeng/env/clip/
cd /DDN_ROOT/wjpeng/dataset/cc12m/

img2dataset --url_list cc12m.tsv --input_format "tsv"\
         --url_col "url" --caption_col "caption" --output_format webdataset \
         --output_folder data --processes_count 32 --thread_count 128 --resize_mode no --incremental
