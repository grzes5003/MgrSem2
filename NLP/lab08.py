import os
from typing import Any, Tuple

import pandas as pd
import requests
import numpy as np

from util import read_files_list


def top_50(path: str):
    files = os.listdir(path)

    sizes = [(file_path, os.stat(f'{path}/{file_path}').st_size)
             for file_path in files]
    sizes.sort(key=lambda item: item[1], reverse=True)

    paths = [f'{path}/{file}' for (file, _) in sizes]
    return list([(file, content) for ((file, _), content) in zip(sizes, list(read_files_list(paths)))])


def chunkwise(t, size=2):
    it = iter(t)
    return list(zip(*[it]*size))


def parse_content(resp: str):
    lines = resp.replace('\t', ' ').split('\n')

    indices = [i for i, x in enumerate(lines) if x == '']
    acc = 0
    for i in range(len(indices)):
        lines.insert(indices[i] + acc, '')
        acc += 1
    chunks = chunkwise(lines)
    parsed, flag = [], True
    for line, desc in chunks:
        if len(line) == 0:
            flag = True
            continue
        word, trait = line.split()
        *base, info, more = desc.split()
        parsed.append((word, " ".join(base), info, trait, 'new' if flag else more))
        if flag:
            flag = False
    return parsed


def lem_and_split(files: [Tuple[str, Any]]):
    """
    Use the lemmatized and sentence split documents
    """
    for file, content in files:
        if os.path.isfile(f'data/tagged/{file}.csv'):
            continue
        print(f'>>> parsing {file}')
        resp = requests.post("http://localhost:9200", data=content.encode("utf-8"))
        data = parse_content(resp.content.decode("utf-8"))
        df = pd.DataFrame(data, columns=['word', 'base', 'info', 'trait', 'more'])
        df.to_csv(f'results/lab08/{file}.csv', index=False)
        print(f'<<< saved')


def exc05(path: str):
    files = os.listdir(path)

    dfs = []
    for file in files:
        df = pd.read_csv(f'{path}/{file}')
        df_words = df[(df['word'].str.istitle()) & (df['more'] != 'new')]
        df_words['idx'] = df_words.index
        groups = [(k, g) for k, g in df_words.groupby(df_words['idx'] - np.arange(df_words.shape[0]))]
        formations = []
        for k, g in groups:
            formations.append((g['base'].str.cat(sep=' ')))
        df_formations = pd.DataFrame(formations, columns=['base'])
        df_tmp = df_formations['base']\
            .value_counts()\
            .to_frame().reset_index()
        dfs.append(df_tmp)
    df_res: pd.DataFrame = pd.concat(dfs).groupby(['index']).sum().reset_index()
    df_res.nlargest(50, 'base').to_csv(f'{path}/exc05_nlargest.csv', index=False)


if __name__ == '__main__':
    path = 'C:/Users/xgg/PycharmProjects/NLP/data/ustawy'
    load_path = 'C:/Users/xgg/PycharmProjects/MgrSem2/NLP/data/tagged'
    res = top_50(path)
    lem_and_split(res)

    # exc05(load_path)
