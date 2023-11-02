import json

import numpy as np
import requests
import os
import inflect

from PIL import Image


if __name__ == '__main__':
    dataset_file = '/DDN_ROOT/wjpeng/dataset/countBench/CountBench.json'
    output_dir = '/DDN_ROOT/wjpeng/dataset/countBench/data'
    timeout = 10
    os.makedirs(output_dir, exist_ok=True)
    p = inflect.engine()

    with open(dataset_file, 'r') as file:
        dataset_raw = json.load(file)

    per_number_cnt = [0] * 11
    for idx, entry in enumerate(dataset_raw):
        image_url = entry['image_url']
        text = entry['text']
        number = entry['number']
        number_word = p.number_to_words(number)

        try:
            raw_image = Image.open(requests.get(image_url, stream=True, timeout=timeout).raw).convert('RGB')
            # 提取文件名
            image_filename = os.path.join(output_dir, f'{number_word}_{per_number_cnt[number]:02d}.jpg')
            ann_filename = os.path.join(output_dir, f'{number_word}_{per_number_cnt[number]:02d}.json')
            per_number_cnt[number] += 1

            # 保存图片
            raw_image.save(image_filename)
            # 保存文本
            with open(ann_filename, 'w') as txt_file:
                ret = {
                    'number': number,
                    'number_word': number_word,
                    'text': text,
                }
                json.dump(ret, txt_file, indent=4)
        except:
            print(f"下载失败：{image_url}")

        print(f'[{idx + 1} / {len(dataset_raw)} ({(idx + 1) / len(dataset_raw) * 100:.2f} %)]')

    print(f'下载完成：'
          f'\ntotal: {np.array(per_number_cnt).sum()}'
          f'\nper number: {per_number_cnt}')
