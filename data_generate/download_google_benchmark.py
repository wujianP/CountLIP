import json

import numpy as np
import requests
import os
import inflect

from PIL import Image


if __name__ == '__main__':
    dataset_file = '/DDN_ROOT/wjpeng/dataset/countBench/google/CountBench.json'
    output_dir = '/DDN_ROOT/wjpeng/dataset/countBench/google/data'
    os.makedirs(output_dir, exist_ok=True)
    p = inflect.engine()
    for i in range(1, 11):
        os.makedirs(os.path.join(output_dir, p.number_to_words(i+1)))

    with open(dataset_file, 'r') as file:
        dataset_raw = json.load(file)

    per_number_cnt = [0] * 11
    for idx, entry in enumerate(dataset_raw):
        image_url = entry['image_url']
        text = entry['text']
        number = entry['number']
        number_word = p.number_to_words(number)

        response = requests.get(image_url)

        if response.status_code == 200:
            # 提取文件名
            image_filename = os.path.join(output_dir, number_word, f'{per_number_cnt[number]:02d}.jpg')
            ann_filename = os.path.join(output_dir, number_word, f'{per_number_cnt[number]:02d}.json')
            per_number_cnt[number] += 1

            # 保存图片
            with open(image_filename, 'wb') as img_file:
                img_file.write(response.content)
            # 保存文本
            with open(ann_filename, 'wb') as txt_file:
                ret = {
                    'number': number,
                    'text': text,
                }
                json.dumps(ret, indent=4)
        else:
            print(f"下载失败：{image_url}")
        print(f'[{idx + 1} / {len(dataset_raw)} ({(idx + 1) / len(dataset_raw) * 100:.2f} %)]')

    print(f'下载完成：'
          f'\ntotal: {np.array(per_number_cnt).sum()}'
          f'\nper number: {per_number_cnt}')
