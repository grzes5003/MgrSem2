import functools
import os
from collections import Counter
from typing import Callable

import spacy
from spacy import Language
from spacy.lang.pl import Polish
from spacy.tokenizer import Tokenizer
from spacy.tokens import Token, Doc
import pandas as pd

import multiprocessing as mp

from util import read_files_dir, read_lines_dir, plot_data, plot_data2, read_lines


def custom_tokenizer(nlp: Language = Polish()) -> Tokenizer:
    special_cases = {"Dz.U.": [{"ORTH": "Dz.U."}]}
    return Tokenizer(nlp.vocab, rules=special_cases)


def word_filter(strings: [str]) -> [str]:
    ...


def is_excluded(token: Token) -> bool:
    return token.is_stop and not token.is_punct and token.is_alpha and len(token.text) >= 2


def get_freq_df(doc: Doc, filter: Callable[[Token], bool] = None) -> pd.DataFrame:
    words = [token.text for token in doc if filter(token)] if filter else [token.text for token in doc]
    word_freq = Counter(words)
    return pd.DataFrame(word_freq.items(), columns=['word', 'freq'])


def concat(*args: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(args).groupby(['word']).sum().reset_index()


def pipe(file_names: [str]):
    print(f"P{mp.current_process()}: starting")
    files = [(file_name, '\n'.join(read_lines(file_name))) for file_name in file_names]

    nlp = spacy.load("pl_core_news_md")
    tokenizer = custom_tokenizer(nlp)
    nlp.tokenizer = tokenizer

    print(f"P{mp.current_process()}: parsing")
    docs = [(file_name, nlp(content)) for file_name, content in files]
    print(f"P{mp.current_process()}: concatenating")
    df = functools.reduce(concat, [get_freq_df(doc[1], is_excluded) for doc in docs])
    print(f"P{mp.current_process()}: done")
    return df


def exc_05():
    ...


if __name__ == '__main__':
    path = 'C:/Users/xgg/PycharmProjects/NLP/data/ustawy'

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

    plot_data2(df_res.sort_values(by='freq', ascending=False)[:50], x='word', y='freq', hue=False, errorbar='ci')
    ...
