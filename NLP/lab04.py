import functools
import math
import os
import multiprocessing as mp
from collections import Counter
from itertools import pairwise
from typing import Tuple, Callable, List

import numpy as np
import pandas as pd
from spacy.lang.pl import Polish
import xml.etree.ElementTree as et

from util import read_files_list


def concat(*args: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(args).groupby(['a', 'b']).sum().reset_index()


def is_excluded(tokens: Tuple[str, str]) -> bool:
    return all([token.isalpha() for token in tokens])


def get_freq_df(tpl: List[Tuple[str, str]], filter: Callable[[Tuple[str, str]], bool] = None) -> pd.DataFrame:
    words = [token for token in tpl if filter(token)] if filter else [token for token in tpl]
    word_freq = Counter(words)
    return pd.DataFrame([(*key, val) for key, val in word_freq.items()], columns=['a', 'b', 'freq'])


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

    p_count = min(mp.cpu_count(), 8)
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


def exc04(df_words: pd.DataFrame, df_bigrams: pd.DataFrame):
    """
    Use pointwise mutual information to compute the measure for all pairs of words.
    """
    all_words = df_words['freq'].sum()
    all_pairs = df_bigrams['freq'].sum()

    df_merged = df_bigrams.merge(df_words, how='left', left_on='a', right_on='word')
    df_merged = df_merged.merge(df_words, how='left', left_on='b', right_on='word')
    df_merged = df_merged.rename(columns={'freq_x': 'freq', 'freq_y': 'freq_a', 'freq': 'freq_b'})
    df_merged.drop(['word_x', 'word_y'], axis=1, inplace=True)

    df_merged['freq_pmi'] = np.log2(
        (df_merged['freq'] / all_pairs) / ((df_merged['freq_a'] * df_merged['freq_b']) / all_words))

    df_merged.to_csv('lab04_pmi_freq.csv', index=False)


def exc05_06(df_pmi: pd.DataFrame):
    """
    Sort the word pairs according to that measure in the descending order and determine top 10 entries.
    """
    print(df_pmi.sort_values(by=['freq_pmi'], ascending=False).head(10))

    """
    Filter bigrams with number of occurrences lower than 5. 
    Determine top 10 entries for the remaining dataset (>=5 occurrences)
    """
    print(df_pmi[df_pmi['freq'] >= 5].sort_values(by=['freq_pmi'], ascending=False).head(10))


def exc10(path: str):
    """
    Compute the same statistics as for the non-lemmatized words
    (i.e. PMI) and print top-10 entries with at least 5 occurrences.
    """

    tree = et.parse(source=path)
    root = tree.getroot()

    tokens = [(tok.find("orth").text, tok.find("lex").find("base").text, tok.find("lex").find("ctag").text)
              for tok in root.findall(".//tok")]

    df_parsed = pd.DataFrame(tokens, columns=['orth', 'base', 'ctag'])
    df_parsed.to_csv('lab04_morf_parsed.csv', index=False)


if __name__ == '__main__':
    path = 'C:/Users/xgg/PycharmProjects/NLP/data/ustawy'
    path_df_bigrams = 'results/lab04/lab04_raw_bigrams.csv'
    path_df_words = 'results/lab04/lab04_words_freq.csv'
    path_df_pmi = 'results/lab04/lab04_pmi_freq.csv'

    # exc01_02_03(path)
    # df_words = pd.read_csv(path_df_words)
    # df_bigrams = pd.read_csv(path_df_bigrams)

    # exc04(df_words, df_bigrams)

    # df_pmi = pd.read_csv(path_df_pmi)
    # exc05_06(df_pmi)

    exc10('results/lab04/lab04_words_corp.xml')
    ...
