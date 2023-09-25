import json
import requests
import os
import inflect

from PIL import Image


if __name__ == '__main__':
    dataset_file = '/DDN_ROOT/wjpeng/dataset/countBench/google/CountBench.json'
    output_dir = '/DDN_ROOT/wjpeng/dataset/countBench/google/data'
    os.makedirs(output_dir, exist_ok=True)
    p = inflect.engine()

    with open(dataset_file, 'r') as file:
        dataset_raw = json.load(file)

    from IPython import embed
    embed()

    for entry in dataset_raw:
        image_url = entry['image_url']
        response = requests.get(image_url)

        if response.status_code == 200:
            # 提取文件名
            filename = os.path.join('images', os.path.basename(image_url))

            # 保存图片
            with open(filename, 'wb') as img_file:
                img_file.write(response.content)
                print(f"下载成功：{filename}")
        else:
            print(f"下载失败，状态码：{response.status_code}")
