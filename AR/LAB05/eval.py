import itertools
from dataclasses import dataclass
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass(repr=True)
class Res:
    time: float
    line: int
    rank: int

    @classmethod
    def from_str(cls, _input: str):
        items = _input.split(';')
        return cls(time=float(items[0][2:]),
                   line=int(items[1][2:]),
                   rank=int(items[2][2:]))


def read_logs(path: str) -> [Res]:
    _start_char = 't'

    with open(path, 'r') as f:
        lines = f.readlines()

    lines = list(itertools.dropwhile(lambda line: line[0] != _start_char, lines))
    lines = [line.replace(' ', '').replace('\n', '') for line in lines]

    return [Res.from_str(line) for line in lines]


def obj2df(results: [Res]) -> pd.DataFrame:
    record = []
    for item in results:
        record.append([item.time, item.line, item.rank])
    return pd.DataFrame(record, columns=['time', 'line', 'rank'])


def exc02(df: pd.DataFrame):
    """
    sprawdz czy czasy różnią się w zależności od numeru linii,
    policz róznicę między największym, a najmniejszym czasem
    """
    diff = df['time'].max() - df['time'].min()
    print(f'{diff}')

    sns.lineplot(x='line', y='time', data=df)
    plt.show()


def exc03(df: pd.DataFrame):
    gdf = df.groupby('rank')
    """
    sprawdz, w jaki sposób linie są przyporządkowywane do procesów
    czy można powiedzieć że jest to podział blokowy albo cykliczny?
    """
    gdf.apply(print)

    """
    porownaj liczbę linii przyporządkowaną do każdego z procesów workerów
    """
    print(gdf.size())

    """
    Policz sumę czasów dla każdego z procesów workerów. 
    Porównaj te czasy do siebie oraz  do czasu działania całego programu.
    """
    print(gdf['time'].sum())


if __name__ == '__main__':
    sns.set_theme(style="darkgrid")

    path_perf = 'results.log'
    res = read_logs(path_perf)
    df = obj2df(res)

    exc03(df)
