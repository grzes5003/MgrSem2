import itertools
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


@dataclass(repr=True)
class Res:
    time: float
    iters: int
    n: int
    cores: int

    @classmethod
    def from_str(cls, _input: str):
        items = _input.split(';')
        return cls(time=float(items[0][2:]),
                   iters=int(items[1][3:]),
                   n=int(items[2][2:]),
                   cores=int(items[3][2:]))


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
        record.append([item.time, item.iters, item.n, item.cores])
    return pd.DataFrame(record, columns=['time', 'iters', 'n', 'cores'])


def plot_eff(df: pd.DataFrame):
    t1 = df[df['cores'] == 1].groupby('n').mean()
    df['speedup'] = t1.loc[df['n']].reset_index()['time'] / df['time']
    df['eff'] = df['speedup'] / df['cores']

    sns.set_theme(style="darkgrid")
    ax = sns.lineplot(x=range(0, 13), y=np.repeat(1, 13), linestyle='--', lw=1)
    sns.pointplot(x='cores', y='eff', data=df, hue='n', errorbar='sd', capsize=.2, ax=ax)

    ax.set(ylabel='Efficiency')
    ax.set_title('Efficiency based on used cores')
    ax.set(xlabel='Number of cores')
    ax.legend(title='Size of problem [n]')

    plt.show()

    sns.set_theme(style="darkgrid")
    ax = sns.pointplot(x="cores", y='speedup', data=df, hue='n', errorbar='sd')
    plt.plot([0, 6], [1, 12], linestyle='--', lw=1)

    ax.set(ylabel='Speedup')
    ax.set_title('Speedup based on used cores')
    ax.set(xlabel='Number of cores')
    ax.legend(title='Size of problem [n]')

    plt.show()


def plot(filepath: str):
    arr = np.load(filepath)
    plt.imshow(arr, origin='upper', cmap='hot')
    plt.colorbar()
    plt.gca().set_aspect('equal')
    plt.savefig(f'results/heatmap.png', dpi=200)
    plt.show()


if __name__ == '__main__':
    # path = 'results/result.npy'
    # plot(path)

    path_perf = 'results/performance.log'
    res = read_logs(path_perf)
    df = obj2df(res)

    plot_eff(df)
    ...
