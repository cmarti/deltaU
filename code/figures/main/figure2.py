from code.plot_utils import (
    FIG_WIDTH,
    add_panel_labels,
    apply_plot_style,
    plot_cv_r2_curves,
)

import matplotlib.pyplot as plt
import pandas as pd
from gpmap.plot.mpl import plot_correlation_U_sites, plot_interaction_matrix
from scipy.stats import pearsonr


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
    r2 = pd.read_csv("results/simulations.r2.csv", index_col=0)
    r2_2 = pd.read_csv("results/simulations.r2.2.csv", index_col=0)
    r2 = pd.concat([r2, r2_2])

    print("Making figure...")
    fig, subplots = plt.subplots(
        2, 2, figsize=(0.55 * FIG_WIDTH, 0.475 * FIG_WIDTH)
    )

    print("  Plotting prior a matrix...")
    axes = subplots[0, 0]
    plot_interaction_matrix(a_matrix, axes,
                            cbar_label="Interaction strength ($1/a_{ij}$)")

    print("  Plotting prior correlation landscape...")
    axes = subplots[0, 1]
    plot_correlation_U_sites(corr, axes, y="cor")

    print("  Plotting prior vs inferred correlation landscape...")
    axes = subplots[1, 0]
    plot_prior_vs_inferred_corr(corr, inferred_corr, axes)

    print("  Plotting R2 vs training set size for model comparison...")
    axes = subplots[1, 1]
    plot_cv_r2_curves(r2, axes)

    print("  Saving figure...")
    fig.tight_layout()
    fig.subplots_adjust(left=0.10, right=0.98, top=0.98, bottom=0.10)
    add_panel_labels(subplots, ["A", "B", "C", "D"], x_offset=-0.22)
    fig.savefig("figures/figure2.png", dpi=300)
    fig.savefig("figures/figure2.svg", dpi=300)
    print("Done.")
