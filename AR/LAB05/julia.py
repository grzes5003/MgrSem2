import functools
import sys

from mpi4py.futures import MPIPoolExecutor
from mpi4py import MPI

import numpy as np


x0, x1, w = -2.0, +2.0, 640 * 2
y0, y1, h = -1.5, +1.5, 480 * 2
dx = (x1 - x0) / w
dy = (y1 - y0) / h

c = complex(0, 0.65)


def timer(func):
    @functools.wraps(func)
    def inner(*args, **kwargs):
        start = MPI.Wtime()
        result = func(*args, **kwargs)
        end = MPI.Wtime()
        print(f't={end - start};k={args[0]};r={MPI.COMM_WORLD.Get_rank()}')
        return result

    return inner


def julia(x, y):
    z = complex(x, y)
    n = 255
    while abs(z) < 3 and n > 1:
        z = z ** 2 + c
        n -= 1
    return n


@timer
def julia_line(k, variant: bool = False):
    if variant and k % 100 == 0:
        arr = np.random.rand(10000, 4000)
        np.fft.fft2(arr)
        return bytearray(w)

    line = bytearray(w)
    y = y1 - k * dy
    for j in range(w):
        x = x0 + j * dx
        line[j] = julia(x, y)
    return line


if __name__ == '__main__':
    variant = True if len(sys.argv) == 2 else False

    start = MPI.Wtime()
    with MPIPoolExecutor() as executor:
        image = executor.map(julia_line, range(h), [variant]*h)
        with open('julia.pgm', 'wb') as f:
            f.write(b'P5 %d %d %d\n' % (w, h, 255))
            for line in image:
                f.write(line)
    end = MPI.Wtime()
    print(f'full_t={end - start}')
