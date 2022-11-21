import functools
import os
import multiprocessing as mp
from collections import Counter
from itertools import pairwise
from typing import Tuple, Callable, List

import pandas as pd
from spacy.lang.pl import Polish

from util import read_files_list


def concat(*args: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(args).groupby(['word']).sum().reset_index()


def is_excluded(tokens: Tuple[str, str]) -> bool:
    return all([token.isalpha() for token in tokens])


def get_freq_df(tpl: List[Tuple[str, str]], filter: Callable[[Tuple[str, str]], bool] = None) -> pd.DataFrame:
    words = [token for token in tpl if filter(token)] if filter else [token for token in tpl]
    word_freq = Counter(words)
    return pd.DataFrame(word_freq.items(), columns=['word', 'freq'])


def pipe(file_names: [str]):
    print(f"P{mp.current_process()}: starting")
    nlp = Polish()
    print(f"P{mp.current_process()}: parsing")

    print(f"P{mp.current_process()}: parsing")
    # docs = [nlp.tokenizer(content) for content in read_files_list(file_names)]
    coll_pairs = []
    for content in read_files_list(file_names):
        pairs = list(pairwise(list(
            map(lambda item: str(item).lower(),
                nlp.tokenizer(content)))))
        coll_pairs.append(pairs)
    # coll_pairs = functools.reduce(list.__add__, coll_pairs)
    print(f"P{mp.current_process()}: concatenating")
    df = functools.reduce(concat, [get_freq_df(coll_pair, is_excluded) for coll_pair in coll_pairs])
    print(f"P{mp.current_process()}: done")
    return df


def exc01_02_03(path: str):
    """
    Use SpaCy tokenizer API to tokenize the text from the law corpus
    &&
    Compute bigram counts of downcased tokens.
    &&
    Discard bigrams containing characters other than letters.
    """

    files = os.listdir(path)
    file_names = [f'{path}/{file}' for file in files]

    p_count = min(mp.cpu_count(), 6)
    n = int(len(file_names) / p_count)
    print(f"{p_count} for {n} each")
    with mp.Pool(processes=p_count) as pool:
        dfs = pool.map(
            pipe,
            [file_names[i:i + n] for i in range(0, len(file_names), n)]
        )

    print(f"concatenating")
    df_res = functools.reduce(concat, dfs)

    df_res.to_csv('lab04_raw_freq.csv', index=False)
    ...


if __name__ == '__main__':
    path = 'C:/Users/xgg/PycharmProjects/NLP/data/ustawy'

    exc01_02_03(path)
