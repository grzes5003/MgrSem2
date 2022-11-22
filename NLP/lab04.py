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
    coll_pairs = []
    for content in read_files_list(file_names):
        pairs = list(pairwise(list(
            map(lambda item: str(item).lower(),
                nlp.tokenizer(content)))))
        coll_pairs.append(pairs)
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


def exc07(path: str):
    """
    Use KRNNT or Clarin-PL API(https://ws.clarin-pl.eu/tager.shtml) to tag and lemmatize the corpus.
    """
    tree = et.parse(source=path)
    root = tree.getroot()

    tokens = [
        (tok.find("orth").text, tok.find("lex").find("base").text, tok.find("lex").find("ctag").text.split(':')[0])
        for tok in root.findall(".//tok")]

    df_parsed = pd.DataFrame(tokens, columns=['orth', 'base', 'cat'])
    df_parsed.to_csv('lab04_morf_parsed_cat.csv', index=False)


def words2morf(df_words: pd.DataFrame, df_morf: pd.DataFrame):
    df_merge = df_words.merge(df_morf, how='left', left_on='word', right_on='orth')
    df_prop = df_merge[['freq']]
    df_prop['base'] = df_merge['base'] + ':' + df_merge['cat']
    df_res: pd.DataFrame = df_prop.groupby(['base']).sum().reset_index()

    df_res.to_csv('results/lab04/lab04_words_freq_morf.csv', index=False)


def exc08_09(df_bigrams: pd.DataFrame, df_morf: pd.DataFrame):
    """
    Using the tagged corpus compute bigram statistic for the tokens containing:
    a. lemmatized, downcased word b. morphosyntactic category of the word (subst, fin, adj, etc.)

    Compute the same statistics as for the non-lemmatized words
    (i.e. PMI) and print top-10 entries with at least 5 occurrences.
    """
    df_merge = df_bigrams.merge(df_morf, how='left', left_on='a', right_on='orth')
    df_merge = df_merge.merge(df_morf, how='left', left_on='b', right_on='orth')

    df_prop = df_merge[['freq']]
    df_prop['base_a'] = df_merge['base_x'] + ':' + df_merge['cat_x']
    df_prop['base_b'] = df_merge['base_y'] + ':' + df_merge['cat_y']
    # df_prop.groupby(['base_a', 'base_b']).sum().reset_index()
    df_res: pd.DataFrame = df_prop.groupby(['base_a', 'base_b'], as_index=False)['freq'].sum()

    """
    Print top-10 entries with at least 5 occurrences.
    """
    print(df_res[df_res['freq'] >= 5].sort_values(by=['freq_pmi'], ascending=False).head(10))

    df_res.to_csv('results/lab04/lab04_bigrams_morf.csv', index=False)


def exc10(df_morf_words: pd.DataFrame, df_morf_biograms: pd.DataFrame):
    """
    Compute the same statistics as for the non-lemmatized words (i.e. PMI)
    """
    all_words = df_morf_words['freq'].sum()
    all_pairs = df_morf_biograms['freq'].sum()

    df_merged = df_morf_biograms.merge(df_morf_words, how='left', left_on='base_a', right_on='base')
    df_merged = df_merged.merge(df_morf_words, how='left', left_on='base_b', right_on='base')
    df_merged = df_merged.rename(columns={'freq_x': 'freq', 'freq_y': 'freq_a', 'freq': 'freq_b'})
    df_merged.drop(['base_x', 'base_y'], axis=1, inplace=True)

    df_merged['freq_pmi'] = np.log2(
        (df_merged['freq'] / all_pairs) / ((df_merged['freq_a'] * df_merged['freq_b']) / all_words))

    """
    Filter bigrams with number of occurrences lower than 5. 
    Determine top 10 entries for the remaining dataset (>=5 occurrences)
    """
    print(df_merged[df_merged['freq'] >= 5].sort_values(by=['freq_pmi'], ascending=False).head(10))

    # df_merged.to_csv('lab04_pmi_morf.csv', index=False)


if __name__ == '__main__':
    path = 'C:/Users/xgg/PycharmProjects/NLP/data/ustawy'
    path_df_bigrams = 'results/lab04/lab04_raw_bigrams.csv'
    path_df_words = 'results/lab04/lab04_words_freq.csv'
    path_df_pmi = 'results/lab04/lab04_pmi_freq.csv'
    path_df_morf = 'results/lab04/lab04_morf_parsed_cat.csv'

    path_df_morf_words = 'results/lab04/lab04_words_freq_morf.csv'
    path_df_diagrams_morf = 'results/lab04/lab04_bigrams_morf.csv'

    # exc01_02_03(path)
    df_words = pd.read_csv(path_df_words)
    # df_bigrams = pd.read_csv(path_df_bigrams)

    # exc04(df_words, df_bigrams)

    # df_pmi = pd.read_csv(path_df_pmi)
    # exc05_06(df_pmi)

    # exc07('results/lab04/lab04_words_corp.xml')
    df_morf = pd.read_csv(path_df_morf)

    # words2morf(df_words, df_morf)
    # exc08_09(df_bigrams, df_morf)

    df_morf_words = pd.read_csv(path_df_morf_words)
    df_diagrams_morf = pd.read_csv(path_df_diagrams_morf)
    exc10(df_morf_words, df_diagrams_morf)
    ...
