import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def plot_data_stack(df: pd.DataFrame, *, hue=True):
    sns.set_theme()
    print(df)
    if hue:
        sns.histplot(x='year', hue='type', weights='number',
                     multiple='stack', data=df, shrink=0.7, palette="ch:.25")
    else:
        sns.histplot(x='year', weights='number', data=df, shrink=0.7, palette="ch:.25")

    plt.xticks(
        rotation=45,
        horizontalalignment='right',
        fontweight='light',
        fontsize='x-small'
    )

    plt.show()


def plot_data(df: pd.DataFrame, *, hue=True):
    sns.set_theme()
    print(df)
    if hue:
        sns.catplot(data=df, x="year", hue='type', y='number', kind="bar", palette="ch:.25", stacked=True)
    else:
        sns.catplot(data=df, x="year", y='number', kind="bar", palette="ch:.25")

    plt.xticks(
        rotation=45,
        horizontalalignment='right',
        fontweight='light',
        fontsize='x-small'
    )

    plt.show()
