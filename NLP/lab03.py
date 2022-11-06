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


def concat(dfs: [pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(dfs).groupby(['word', 'freq']).sum().reset_index()


def pipe(file_names: [str]):
    files = [(file_name, '\n'.join(read_lines(file_name))) for file_name in file_names]

    nlp = spacy.load("pl_core_news_md")
    tokenizer = custom_tokenizer(nlp)
    nlp.tokenizer = tokenizer

    docs = [(file_name, nlp(content)) for file_name, content in files]
    dfs = [get_freq_df(doc[1], is_excluded) for doc in docs]
    df = concat(dfs)
    return df


def exc_05():
    ...


if __name__ == '__main__':
    path = 'C:/Users/xgg/PycharmProjects/NLP/data/ustawy'

    files = os.listdir(path)[:20]
    file_names = [f'{path}/{file}' for file in files]

    # [(file_name, '\n'.join(content)) for file_name, content in read_lines_dir(path)]

    # nlp = spacy.load("pl_core_news_md")
    # tokenizer = custom_tokenizer(nlp)
    # nlp.tokenizer = tokenizer

    p_cout = min(mp.cpu_count(), 4)
    n = int(len(file_names)/p_cout)
    with mp.Pool(processes=p_cout) as pool:
        df = pool.map(
            pipe,
            [file_names[i:i + n] for i in range(0, len(file_names), n)]
        )

    # docs = [(file_name, nlp(content)) for file_name, content in files]
    #
    # dfs = [get_freq_df(doc[1], is_excluded) for doc in docs]
    # df = concat(dfs)

    plot_data2(df.sort_values(by='freq', ascending=False)[:50], x='word', y='freq', hue=False, errorbar='ci')

    # common_words = word_freq.most_common(5)

    ...
