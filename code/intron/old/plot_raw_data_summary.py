import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import pearsonr


if __name__ == "__main__":
    np.random.seed(0)
    fpath = "data/raw/intron.csv"
    data = pd.read_csv(fpath, index_col=0).dropna()
    print(data)
    # exit()

    bins = np.linspace(-6, 3, 50)
    fig, subplots = plt.subplots(1, 2, figsize=(6, 3))

    x, y = "DimSum fitness (30ºC) ", "DimSum fitness (37ºC) "
    x, y = "Log2 fold-change (30ºC)", "Log2 fold-change (37ºC)"
    axes = subplots[0]
    sns.histplot(
        x=data[x],
        y=data[y],
        cmap="binary",
        bins=[bins, bins],
        ax=axes,
    )
    r = pearsonr(data[x], data[y])[0]
    axes.text(
        0.05,
        0.95,
        r"$\rho$" + f"={r:.2f}",
        transform=axes.transAxes,
        ha="left",
        va="top",
    )
    axes.axline((0, 0), slope=1, lw=0.5, linestyle="--", c="grey")
    axes.axvline(0, lw=0.5, linestyle="--", c="grey")
    axes.axhline(0, lw=0.5, linestyle="--", c="grey")
    axes.set(ylabel="30C $y$", xlabel="37C $y$")

    # axes = subplots[1]
    # sns.histplot(
    #     x=np.log(data["30C_y_var"]),
    #     y=np.log(data["37C_y_var"]),
    #     cmap="binary",
    #     bins=[bins, bins],
    #     ax=axes,
    # )
    # r = pearsonr(np.log(data["30C_y_var"]), np.log(data["37C_y_var"]))[0]
    # axes.text(
    #     0.05,
    #     0.95,
    #     r"$\rho$" + f"={r:.2f}",
    #     transform=axes.transAxes,
    #     ha="left",
    #     va="top",
    # )
    # axes.axline((0, 0), slope=1, lw=0.5, linestyle="--", c="grey")
    # axes.axvline(0, lw=0.5, linestyle="--", c="grey")
    # axes.axhline(0, lw=0.5, linestyle="--", c="grey")
    # axes.set(ylabel="30C $\log(y_{var})$", xlabel="37C $\log(y_{var})$")

    fig.tight_layout()
    fig.savefig("figures/conditions.raw.png", dpi=300)
