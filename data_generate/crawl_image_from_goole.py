"""
This script is used to download images from the internet
"""

import requests
from bs4 import BeautifulSoup
import os


def download_images(query, num_images, save_dir):
    # 构建Google图片搜索的URL，其中q参数为搜索关键词
    url = f'https://www.google.com/search?q={query}&tbm=isch'

    # 设置请求头，模拟浏览器访问
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.81 Safari/537.36'}
    response = requests.get(url, headers=headers)  # 发送HTTP请求获取网页内容
    soup = BeautifulSoup(response.text, 'html.parser')  # 使用Beautiful Soup解析网页

    # 找到所有图片的img标签
    image_tags = soup.find_all('img', {'class': 't0fcAb'})

    os.makedirs(save_dir, exist_ok=True)  # 创建保存目录

    from IPython import embed
    embed()
    # 循环下载图片
    for i, img_tag in enumerate(image_tags[:num_images]):
        img_url = img_tag['src']  # 提取图片URL
        img_data = requests.get(img_url).content  # 发送请求获取图片数据
        with open(os.path.join(save_dir, f'{query}_{i + 1}.jpg'), 'wb') as f:  # 保存图片数据到文件
            f.write(img_data)


if __name__ == '__main__':
    keyword = 'cats'
    num_images_per_category = 10
    save_dir = '/DDN_ROOT/wjpeng/dataset/countBench'
    print('Begin')
    download_images(keyword, num_images_per_category, save_dir)
    print('Finish')
