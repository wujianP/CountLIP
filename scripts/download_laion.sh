# https://github.com/rom1504/img2dataset/blob/main/dataset_examples/laion400m.md

conda activate /discobox/wjpeng/env/clip/
cd /DDN_ROOT/wjpeng/dataset/LAION400M/

# the whole dataset on NGC 5146328
img2dataset --url_list metadata --input_format "parquet"\
         --url_col "URL" --caption_col "TEXT" --output_format webdataset\
           --output_folder laion400m-data --processes_count 16 --thread_count 128 --resize_mode no \
             --save_additional_columns '["NSFW","similarity","LICENSE"]'

# download a subset on NGC 5146327
img2dataset --url_list metadata-00-01 --input_format "parquet"\
         --url_col "URL" --caption_col "TEXT" --output_format webdataset\
           --output_folder laion400m-data-00-01 --processes_count 16 --thread_count 128 --resize_mode no \
             --save_additional_columns '["NSFW","similarity","LICENSE"]'
