import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import pearsonr


if __name__ == "__main__":
    np.random.seed(0)
    fpath = "data/processed/intron.csv"
    data = pd.read_csv(fpath, index_col=0)
    print(data)

    bins = np.linspace(-5, 2, 50)
    fig, subplots = plt.subplots(1, 3, figsize=(9, 3))

    axes = subplots[0]
    sns.histplot(
        x=data["perc_dev_explained"],
        bins=30,
        ax=axes,
    )
    axes.set(xlabel="% deviance explained", ylabel="# sequences", xlim=(0, 1))

    data = data.loc[data["perc_dev_explained"] > 0.0, :]
    print(data.shape)

    axes = subplots[1]
    sns.histplot(
        x=data["30C_y"],
        y=data["37C_y"],
        cmap="binary",
        bins=[bins, bins],
        ax=axes,
    )
    r = pearsonr(data["30C_y"], data["37C_y"])[0]
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

    axes = subplots[2]
    bins = np.linspace(-2, -1, 50)
    sns.histplot(
        x=np.log(data["30C_y_var"]),
        y=np.log(data["37C_y_var"]),
        cmap="binary",
        bins=[bins, bins],
        ax=axes,
    )
    r = pearsonr(np.log(data["30C_y_var"]), np.log(data["37C_y_var"]))[0]
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
    axes.set(ylabel="30C $\log(y_{var})$", xlabel="37C $\log(y_{var})$")

    fig.tight_layout()
    fig.savefig("figures/conditions.png", dpi=300)
