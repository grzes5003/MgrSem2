import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def avg(df: pd.DataFrame):
    df['MA'] = df['reward_sum'].rolling(window=100).mean()
    sns.lineplot(x='idx', y='MA', data=df)

    plt.show()


if __name__ == '__main__':
    sns.set_theme(style="darkgrid")

    paths = ['results/result_01.csv',
             'results/result_02.csv',
             'results/result_03.csv',
             'results/result_04.csv']

    dfs = [pd.read_csv(path) for path in paths]

    avg(dfs[0])
