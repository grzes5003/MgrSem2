import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def avg(df: pd.DataFrame, title: str):
    df['MA'] = df['reward_sum'].rolling(window=100).mean()
    sns.lineplot(x='idx', y='MA', data=df)

    plt.title(f"moving average: {title}")
    plt.show()


def sd(df_path: str, title: str):
    dfs = [(pd.read_csv(f"{df_path}_{idx}.csv"), idx) for idx in range(1, 5)]
    # for df, idx in dfs:
    #     df['hue'] = idx
    df = pd.concat([df for (df, _) in dfs], ignore_index=True)
    df['reward_sum'] = df['reward_sum'].rolling(window=25).mean()
    # df = df.iloc[::10, :]

    sns.lineplot(x='idx', y='reward_sum', data=df)

    plt.title(f"sd: {title}")
    plt.show()


if __name__ == '__main__':
    sns.set_theme(style="darkgrid")

    paths = ['results/result_01',
             'results/result_02',
             'results/result_03',
             'results/result_04']

    # dfs = [pd.read_csv(path) for path in paths]

    # avg(dfs[3], "result 4")
    sd(paths[3], "result 4")
