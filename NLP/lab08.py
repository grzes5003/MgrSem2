import os
from util import read_files_list


def top_50(path: str):
    files = os.listdir(path)

    sizes = [(file_path, os.stat(f'{path}/{file_path}').st_size)
            for file_path in files]
    sizes.sort(key=lambda item: item[1], reverse=True)

    return [(file, read_files_list(f'{path}/{file}')) for (file, _) in sizes[:50]]


if __name__ == '__main__':
    path = 'C:/Users/xgg/PycharmProjects/NLP/data/ustawy'
    res = top_50(path)
    print(res)
