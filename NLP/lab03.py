import functools
import os
import time
from collections import Counter
from typing import Callable

import Levenshtein
import morfeusz2
from spacy import Language
from spacy.lang.pl import Polish
from spacy.tokenizer import Tokenizer
from spacy.tokens import Token, Doc
import pandas as pd

from pathlib import Path
import multiprocessing as mp

from util import plot_data2, read_files_list, setup_es, load_docs, client, load_file_to_doc


def custom_tokenizer(nlp: Language = Polish()) -> Tokenizer:
    special_cases = {"Dz.U.": [{"ORTH": "Dz.U."}]}
    return Tokenizer(nlp.vocab, rules=special_cases)


def word_filter(strings: [str]) -> [str]:
    ...


def is_excluded(token: Token) -> bool:
    return token.is_alpha and len(token.text) >= 2


def get_freq_df(doc: Doc, filter: Callable[[Token], bool] = None) -> pd.DataFrame:
    words = [token.text for token in doc if filter(token)] if filter else [token.text for token in doc]
    word_freq = Counter(words)
    return pd.DataFrame(word_freq.items(), columns=['word', 'freq'])


def concat(*args: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(args).groupby(['word']).sum().reset_index()


def pipe(file_names: [str]):
    print(f"P{mp.current_process()}: starting")
    # nlp = spacy.load("pl_core_news_md")
    nlp = Polish()
    # tokenizer = custom_tokenizer(nlp)
    # nlp.tokenizer = tokenizer

    docs = []
    print(f"P{mp.current_process()}: parsing")
    for content in read_files_list(file_names):
        docs.append(nlp.tokenizer(content))
    print(f"P{mp.current_process()}: concatenating")
    df = functools.reduce(concat, [get_freq_df(doc, is_excluded) for doc in docs])
    print(f"P{mp.current_process()}: done")
    return df


def exc_05(path: str):
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

    df_res.to_csv('lab03_freq.csv', index=False)


def exc_07(df: pd.DataFrame):
    plot_data2(df.sort_values(by='freq', ascending=False)[:50], x='word', y='freq', hue=False, errorbar='ci')


def exc_08_09_10(df: pd.DataFrame):
    """"
    8) find all words that do not appear in that dictionary
    """
    morf = morfeusz2.Morfeusz()
    df_result = df[df.apply(lambda row: morf.analyse(str(row['word']))[0][2][2] == "ign", axis=1)]

    """
    9) Find 30 words with the highest ranks that do not belong to the dictionary
    """
    df_top = df_result.sort_values(by='freq', ascending=False)[:30]
    df_top.to_csv('results/lab03/lab03_freq_top.csv', index=False)

    """
    10) Find 30 random words (i.e. shuffle the words) with 5 occurrences that do not belong to the dictionary
    """
    df_rnd = df_result[df['freq'] == 5].sample(frac=1)[:30]
    df_rnd.to_csv('results/lab03/lab03_freq_rand.csv', index=False)

    return df_result, df_top, df_rnd


def exc_10(df_all: pd.DataFrame, df_nondict: pd.DataFrame, df_top: pd.DataFrame, df_rnd: pd.DataFrame):
    """
    Use Levenshtein distance and the frequency list, to determine the most probable correction
    of the words from lists defined in points 8 and 9. (Note: You don't have to apply the
    distance directly. Any method that is more efficient than scanning the dictionary
    will be appreciated.)
    """
    df_dict = df_all.merge(df_nondict.drop_duplicates(), on=['word', 'freq'], how='left', indicator=True)
    df_dict = df_dict[df_dict['_merge'] == 'left_only']

    autocorr_top = []
    autocorr_rnd = []
    tic = time.perf_counter()
    for word_top, word_rnd in zip(df_top['word'], df_rnd['word']):
        autocorr_top.append(min([(Levenshtein.distance(word_top, dict_word), word_top, dict_word) for dict_word in df_dict['word']],
                            key=lambda item: item[0]))
        autocorr_rnd.append(
            min([(Levenshtein.distance(word_rnd, dict_word), word_rnd, dict_word) for dict_word in df_dict['word']],
                key=lambda item: item[0]))
    toc = time.perf_counter()
    print(f"time={toc - tic:0.4f}")

    df_sug_top = pd.DataFrame(autocorr_top, columns=['dist', 'org', 'sug'])
    df_sug_rng = pd.DataFrame(autocorr_rnd, columns=['dist', 'org', 'sug'])

    print(df_sug_top)
    print(df_sug_rng)


if __name__ == '__main__':
    path = 'C:/Users/xgg/PycharmProjects/NLP/data/ustawy'
    results_filepath = "results/lab03/lab03_freq.csv"

    results_file = Path("results/lab03/lab03_freq.csv")
    if not results_file.is_file():
        exc_05(path)

    # df = pd.read_csv(results_filepath)
    # exc_07(df)

    # df_nondict, df_top, df_rnd = exc_08_09_10(df)

    # exc_10(df, df_nondict, df_top, df_rnd)

    setup_es()
    load_docs(path)
    load_file_to_doc('data/sgjp/sgjp-20221030.tab', name='sgjp')

    print('done')

