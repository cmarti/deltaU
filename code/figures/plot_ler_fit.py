import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import pearsonr
import gpmap.plot.mpl as mplot
from gpmap.space import SequenceSpace


if __name__ == "__main__":
    nodes_df = pd.read_csv(
        "results/intron.corrs.csv", dtype={"seq": str}, index_col="seq"
    )
    print(nodes_df)
    space = SequenceSpace(X=nodes_df.index.values, y=nodes_df["emp_cor"].values)
    edges_df = space.get_edges_df()
    m = pd.read_csv("results/intron.interaction_strength.csv", index_col=0)
    # pred = pd.read_csv("results/intron.ler.landscape.csv", index_col=0)
    # test = pd.read_csv("data/processed/intron.test.csv", index_col=0)
    # test = test.join(pred)

    fig, subplots = plt.subplots(2, 2, figsize=(7, 6))
    subplots = subplots.flatten()

    axes = subplots[0]
    axes.scatter(
        nodes_df["d_jittered"], nodes_df["emp_cor"], color="black", s=10, alpha=0.5, lw=0
    )
    mplot.plot_edges(
        axes, nodes_df, edges_df, x="d_jittered", y="emp_cor", alpha=0.1, color="black"
    )
    axes.axhline(0, color="grey", linestyle="--", lw=0.75, alpha=0.5)
    axes.set(
        xlabel="Hamming distance",
        ylabel="Empirical correlation",
        xticks=np.arange(9),
    )

    axes = subplots[1]
    axes.scatter(
        nodes_df["pred_cor"], nodes_df["emp_cor"], color="black", s=5, alpha=0.5, lw=0
    )
    r2 = pearsonr(nodes_df["pred_cor"], nodes_df["emp_cor"])[0] ** 2
    axes.text(
        0.05,
        0.95,
        r"$R^2$" + f"={r2:.2f}",
        transform=axes.transAxes,
        ha="left",
        va="top",
    )
    axes.set(
        xlabel="Predicted correlation",  # yticklabels=[],
        ylabel="Empirical correlation",
    )
    axes.axline(
        (0, 0), slope=1, color="grey", linestyle="--", lw=0.75, alpha=0.5
    )
    axes.axhline(0, color="grey", linestyle="--", lw=0.75, alpha=0.5)
    axes.axvline(0, color="grey", linestyle="--", lw=0.75, alpha=0.5)

    axes = subplots[2]
    im = axes.imshow(m, cmap="binary")
    labels = m.columns
    axes.set(
        xlabel="Site 1",
        ylabel="Site 2",
        xticks=np.arange(8),
        yticks=np.arange(8),
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.colorbar(
        im,
        ax=axes,
        fraction=0.046,
        pad=0.04,
        label="Interaction strength ($1/a_{ij}$)",
    )

    # axes = subplots[3]
    # axes.scatter(test["f"], test["30C_y"], s=5, lw=0, c="black", alpha=0.5)
    # r2 = pearsonr(test["f"], test["30C_y"])[0] ** 2
    # axes.text(
    #     0.05,
    #     0.95,
    #     r"$R^2$" + f"={r2:.2f}",
    #     transform=axes.transAxes,
    #     ha="left",
    #     va="top",
    # )
    # lims = (-9, 2)
    # axes.set(
    #     xlabel="Predicted fitness",
    #     ylabel="Measured fitness",
    #     xlim=lims,
    #     ylim=lims,
    # )
    # axes.axline(
    #     (0, 0), slope=1, color="grey", linestyle="--", lw=0.75, alpha=0.5
    # )
    # axes.axhline(0, color="grey", linestyle="--", lw=0.75, alpha=0.5)
    # axes.axvline(0, color="grey", linestyle="--", lw=0.75, alpha=0.5)

    fig.tight_layout()
    fig.savefig("figures/intron.ler.fit.png", dpi=300)
