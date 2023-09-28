"""https://github.com/rom1504/img2dataset/blob/main/dataset_examples/SBUcaptions.md"""

conda activate /discobox/wjpeng/env/clip/
cd /DDN_ROOT/wjpeng/dataset/sub_caption/

img2dataset --url_list sbu-captions-all.json --input_format "json" --url_col "image_urls" \
--caption_col "captions" --output_format webdataset --output_folder data \
--processes_count 16 --thread_count 64 --resize_mode no
