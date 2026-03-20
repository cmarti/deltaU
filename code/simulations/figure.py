import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import pearsonr
from code.plot_utils import (
    plot_correlation_landscape,
    plot_interaction_matrix,
    apply_plot_style,
    plot_cv_r2_curves,
    add_panel_labels,
    FIG_WIDTH,
)


def plot_prior_vs_inferred_corr(corr, inferred_corr, axes):
    axes.scatter(
        inferred_corr["pred_cor"],
        corr["cor"],
        color="black",
        s=5,
        alpha=0.5,
        lw=0,
    )
    r2 = pearsonr(inferred_corr["pred_cor"], corr["cor"])[0] ** 2
    axes.text(
        0.05,
        0.95,
        r"$R^2$" + f"={r2:.2f}",
        transform=axes.transAxes,
        ha="left",
        va="top",
        fontsize=8,
    )
    axes.set(
        xlabel="Predicted correlation",  # yticklabels=[],
        ylabel="True prior correlation",
    )
    axes.axline(
        (0, 0), slope=1, color="grey", linestyle="--", lw=0.75, alpha=0.5
    )
    axes.axhline(0, color="grey", linestyle="--", lw=0.75, alpha=0.5)
    axes.axvline(0, color="grey", linestyle="--", lw=0.75, alpha=0.5)


def plot_inferred_vs_true_test(test, axes):
    # axes.scatter(test["fpred"], test["f"], s=5, lw=0, c="black", alpha=0.5,
    #              rasterized=True)
    bins = np.linspace(-7.5, 7.5, 51)
    h, xedges, yedges = np.histogram2d(
        test["fpred"].values,
        test["f"].values,
        bins=bins,
    )
    im = axes.imshow(
        h.T,
        origin="lower",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        aspect="auto",
        cmap="binary",
    )
    plt.colorbar(im, ax=axes, fraction=0.046, pad=0.04, label="# sequences")

    r2 = pearsonr(test["fpred"], test["f"])[0] ** 2
    axes.text(
        0.05,
        0.95,
        r"$R^2$" + f"={r2:.2f}",
        transform=axes.transAxes,
        ha="left",
        va="top",
        fontsize=8,
    )
    axes.set(
        xlabel="Test predicted fitness",
        ylabel="Test true fitness",
        xlim=(-7.5, 7.5),
        ylim=(-7.5, 7.5),
        aspect="equal",
    )
    axes.axline(
        (0, 0), slope=1, color="grey", linestyle="--", lw=0.75, alpha=0.5
    )
    axes.axhline(0, color="grey", linestyle="--", lw=0.75, alpha=0.5)
    axes.axvline(0, color="grey", linestyle="--", lw=0.75, alpha=0.5)


if __name__ == "__main__":
    apply_plot_style()

    print("Loading data...")
    a_matrix = pd.read_csv("results/simulations.prior_a.csv", index_col=0)
    corr = pd.read_csv(
        "results/simulations.prior_correlations.csv", dtype={"seq": str}
    ).set_index("seq")
    inferred_corr = pd.read_csv(
        "results/simulations.corrs.csv", dtype={"seq": str}
    ).set_index("seq")
    r2 = pd.read_csv("results/simulations.r2.csv")

    print("Making figure...")
    fig, subplots = plt.subplots(
        2, 2, figsize=(0.55 * FIG_WIDTH, 0.475 * FIG_WIDTH)
    )

    print("  Plotting prior a matrix...")
    axes = subplots[0, 0]
    plot_interaction_matrix(a_matrix, axes)

    print("  Plotting prior correlation landscape...")
    axes = subplots[0, 1]
    plot_correlation_landscape(corr, axes, y="cor")

    print("  Plotting prior vs inferred correlation landscape...")
    axes = subplots[1, 0]
    plot_prior_vs_inferred_corr(corr, inferred_corr, axes)

    print("  Plotting R2 vs training set size for model comparison...")
    axes = subplots[1, 1]
    plot_cv_r2_curves(axes, r2)

    print("  Saving figure...")
    fig.tight_layout()
    fig.subplots_adjust(left=0.10, right=0.98, top=0.98, bottom=0.10)
    add_panel_labels(subplots, ["A", "B", "C", "D"], x_offset=-0.22)
    fig.savefig("figures/simulations_figure.png", dpi=300)
    fig.savefig("figures/simulations_figure.svg", dpi=300)
    print("Done.")
