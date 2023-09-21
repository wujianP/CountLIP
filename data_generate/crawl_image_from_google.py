"""
This script is used to download images from the internet
"""
import os

import simple_image_download as simp


if __name__ == '__main__':
    categories = ['people', 'cars', 'airplanes', 'candles', 'chairs', 'cats', 'dogs',
                  'books', 'birds', 'horses', 'sheep', 'cow', 'elephants', 'bears',
                  'zebras', 'giraffes', 'ties', 'sports balls', 'kites', 'bottles',
                  'wine glasses', 'forks', 'knives', 'spoons', 'bowls', 'bananas',
                  'apples', 'oranges', 'carrots', 'donuts', 'potted plants', 'trees',
                  'computer monitors', 'vases', 'teddy bear']
    numbers = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten']
    root_dir = '/DDN_ROOT/wjpeng/dataset/countBench/raw'

    my_downloader = simp.Downloader()

    for num in numbers:
        for cat in categories:

            query = f'{num} {cat}'
            print(f'#########################  Begin downloading {query}  ##########################')

            os.makedirs(os.path.join(root_dir, num), exist_ok=True)
            cur_dir = os.path.join(root_dir, num, cat.replace(' ', '_'))
            os.makedirs(cur_dir, exist_ok=True)

            my_downloader.directory = os.path.join(root_dir, num)

            my_downloader.download(keywords=query, limit=50, download_cache=True)

            my_downloader.flush_cache()
