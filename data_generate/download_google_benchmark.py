import json
import os


if __name__ == '__main__':
    from IPython import embed
    embed()
    dataset_file = '/DDN_ROOT/wjpeng/dataset/countBench/google/CountBench.json'
    with open(dataset_file, 'r') as file:
        dataset_raw = json.load(file)
