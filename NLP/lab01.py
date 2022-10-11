import re
from util import read_lines_dir, plot_data_stack, plot_data
import pandas as pd


def counter(lines: [str], regx: str) -> int:
    res = list(filter(lambda arr: len(arr) != 0, [re.findall(regx, line) for line in lines]))
    return len(res)


def parse_year(filename: str) -> str:
    return filename[:4]


def exc01(files: list[tuple[str, list[str]]]):
    regxp = {
        'add': r'((dodaje się ust. )+(0|[1-9][0-9]*))',
        'remove': r'(w ((ust.*)|(art.*)|(pkt*))(?=skreśla się))',
        'amend': r'(art. (0|[1-9][0-9]*) otrzymuje brzmienie)'
    }

    result = {}
    for r in files:
        for expr in regxp:
            key = parse_year(r[0]) + expr
            if key not in result:
                result[key] = 0
            result[key] += counter(r[1], regxp[expr])

    df = pd.DataFrame([(key[:4], key[4:], result[key]) for key in result], columns=['year', 'type', 'number'])

    # calculate percentage
    df['number'] = df['number'] / df.groupby('year').transform('sum')['number']

    plot_data_stack(df)


def exc02(files: list[tuple[str, list[str]]]) -> dict:
    regxp = r'(?i)\b(ustawa|ustawy|ustawie|ustawę|ustawą|ustawo)\b'

    result = {}
    # for r in files:
    #     result[r[0]] = counter(r[1], regxp)
    for r in files:
        if (key := parse_year(r[0])) not in result:
            result[key] = 0
        result[key] += counter(r[1], regxp)

    return result


def exc03(files: list[tuple[str, list[str]]]) -> dict:
    regxp = r'(?i)\b(ustawa|ustawy|ustawie|ustawę|ustawą|ustawo)\b(?=[ \t]+z[ \t]+dnia)'

    result = {}
    # for r in files:
    #     result[r[0]] = counter(r[1], regxp)
    for r in files:
        if (key := parse_year(r[0])) not in result:
            result[key] = 0
        result[key] += counter(r[1], regxp)

    return result


def exc04(files: list[tuple[str, list[str]]]) -> dict:
    regxp = r'(?i)\b(ustawa|ustawy|ustawie|ustawę|ustawą|ustawo)\b(?![ \t]+z[ \t]+dnia)'

    result = {}
    # for r in files:
    #     result[r[0]] = counter(r[1], regxp)
    for r in files:
        if (key := parse_year(r[0])) not in result:
            result[key] = 0
        result[key] += counter(r[1], regxp)

    return result


def exc05(files: list[tuple[str, list[str]]]):
    regxp = r'(?i)(?<!o zmianie )\b(ustawa|ustawy|ustawie|ustawę|ustawą|ustawo)\b'

    result = {}
    for r in files:
        if (key := parse_year(r[0])) not in result:
            result[key] = 0
        result[key] += counter(r[1], regxp)

    df = pd.DataFrame(result.items(), columns=['year', 'number'])

    plot_data(df, hue=False)


if __name__ == '__main__':
    path = 'C:/Users/xgg/PycharmProjects/NLP/data/ustawy'

    lines = read_lines_dir(path)

    # exc01(lines)
    # _all = exc02(lines)
    # _with = exc03(lines)
    # _without = exc04(lines)
    #
    # df_all = pd.DataFrame(_all.items(), columns=['year', 'number'])
    # df_with = pd.DataFrame(_with.items(), columns=['year', 'number'])
    # df_without = pd.DataFrame(_without.items(), columns=['year', 'number'])
    #
    # df_with['type'] = 'with'
    # df_without['type'] = 'without'
    #
    # df_comb = pd.concat([df_with, df_without])
    #
    # equal = [_with[key] + _without[key] == _all[key] for key in _all]
    # print(all(equal))
    # print(f'{equal.count(True)} out of {len(equal)}')
    #
    # plot_data(df_all, hue=False)
    # plot_data_stack(df_comb)

    exc05(lines)
