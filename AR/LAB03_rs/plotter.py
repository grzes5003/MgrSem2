import itertools
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt


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
    a = read_logs(path_perf)
    ...
