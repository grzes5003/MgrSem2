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


def plot_data(df: pd.DataFrame, *, hue=True,
              x='year', y='number',
              errorbar: str = None):
    sns.set_theme()

    print(df)
    if hue:
        sns.catplot(data=df, x=x, hue='type', y=y, kind="bar",
                         palette="ch:.25", stacked=True, errorbar=errorbar)
    else:
        sns.catplot(data=df, x=x, y=y, kind="bar", palette="ch:.25", errorbar=errorbar)

    plt.xticks(
        rotation=45,
        horizontalalignment='right',
        fontweight='light',
        fontsize='x-small'
    )

    plt.show()


def plot_data2(df: pd.DataFrame, *, hue=True,
              x='year', y='number', errorbar: str = None):
    sns.set_theme()

    print(df)
    if hue:
        sns.barplot(data=df, x=x, hue='type', y=y, palette="ch:.25", stacked=True)
    else:
        ax = sns.barplot(data=df, x=x, y=y, palette="ch:.25", errorbar=errorbar)

    ax.set_yscale("log")

    plt.xticks(
        rotation=45,
        horizontalalignment='right',
        fontweight='light',
        fontsize='x-small'
    )

    plt.show()
