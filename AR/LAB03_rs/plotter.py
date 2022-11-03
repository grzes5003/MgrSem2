import numpy as np
import matplotlib.pyplot as plt


def plot(filepath: str):
    arr = np.load(filepath)
    plt.imshow(arr, origin='upper', cmap='hot')
    plt.colorbar()
    plt.gca().set_aspect('equal')
    plt.savefig(f'results/heatmap.png', dpi=200)
    plt.show()


if __name__ == '__main__':
    path = 'results/result.npy'
    plot(path)
