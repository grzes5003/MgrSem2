import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    T = np.load('results/result.npy')
    plt.imshow(T, origin='upper', cmap='hot')
    plt.colorbar()
    plt.gca().set_aspect('equal')
    plt.savefig(f'results/heatmap.png', dpi=150)
