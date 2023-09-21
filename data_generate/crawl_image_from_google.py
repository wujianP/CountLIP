"""
This script is used to download images from the internet
"""
import os

import simple_image_download as simp

if __name__ == '__main__':
    categories_plural = ['people', 'cars', 'airplanes', 'candles', 'chairs', 'cats', 'dogs',
                         'books', 'birds', 'horses', 'sheep', 'cow', 'elephants', 'bears',
                         'zebras', 'giraffes', 'ties', 'sports balls', 'kites', 'bottles',
                         'wine glasses', 'forks', 'knives', 'spoons', 'bowls', 'bananas',
                         'apples', 'oranges', 'carrots', 'donuts', 'potted plants', 'trees',
                         'computer monitors', 'vases', 'teddy bear']
    categories_singular = ['people', 'car', 'airplane', 'candle', 'chair', 'cat', 'dog',
                           'book', 'bird', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                           'zebra', 'giraffe', 'tie', 'sports ball', 'kite', 'bottle',
                           'wine glass', 'fork', 'knife', 'spoon', 'bowl', 'banana',
                           'apple', 'orange', 'carrot', 'donut', 'potted plant', 'tree',
                           'computer monitor', 'vase', 'teddy bear']
    numbers = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten']
    root_dir = '/DDN_ROOT/wjpeng/dataset/countBench/raw'

    my_downloader = simp.Downloader()

    for num in numbers:
        if num == 'one':
            categories = categories_singular
        else:
            categories = categories_plural

        for cat in categories:
            query = f'a photo of {num} {cat}'
            print(f'#########################  Begin downloading {query}  ##########################')

            os.makedirs(os.path.join(root_dir, num), exist_ok=True)
            cur_dir = os.path.join(root_dir, num, cat.replace(' ', '_'))
            os.makedirs(cur_dir, exist_ok=True)

            my_downloader.directory = os.path.join(root_dir, cur_dir)

            query = query.replace(' ', '+')
            my_downloader.download(keywords=query, limit=100)

            my_downloader.flush_cache()
