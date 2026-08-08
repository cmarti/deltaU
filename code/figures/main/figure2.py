from code.plot_utils import (
    FIG_WIDTH,
    add_panel_labels,
    add_r2_label,
    apply_plot_style,
    plot_cv_r2_curves,
    plot_train_pred_comparison,
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gpmap.plot.mpl import plot_correlation_U_sites, plot_interaction_matrix
from scipy.stats import pearsonr


def plot_prior_vs_inferred_corr(corr, inferred_corr, axes):
    axes.scatter(
        inferred_corr["pred_cor"],
        corr["cor"],
        color="black",
        s=3.5,
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
        fontsize=6,
    )
    axes.set(
        xlabel="Predicted correlation",  # yticklabels=[],
        ylabel="True prior correlation",
    )
    axes.axline(
        (0, 0), slope=1, color="grey", linestyle="--", lw=0.5, alpha=0.5
    )
    axes.axhline(0, color="grey", linestyle="--", lw=0.5, alpha=0.5)
    axes.axvline(0, color="grey", linestyle="--", lw=0.5, alpha=0.5)


def plot_test_pred_comparison(test, axes, lims):
    x = test["f"]
    y, yerr = test["fpred"], 2 * np.sqrt(test["f_std"])
    r2 = pearsonr(y, x)[0] ** 2

    axes.errorbar(
        x,
        y,
        yerr=yerr,
        color="grey",
        fmt="none",
        alpha=0.2,
        markeredgewidth=0,
        markersize=2.5,
        lw=0.5,
        zorder=1,
    )

    axes.scatter(x, y, color="black", alpha=0.5, s=2, lw=0, zorder=2)
    axes.set(
        xlim=lims,
        ylim=lims,
        xlabel="True fitness",
        ylabel="Predicted fitness",
    )
    axes.axline((0, 0), (1, 1), lw=0.5, c="grey", linestyle="--")
    add_r2_label(axes, r2)
    axes.text(
        0.95,
        0.05,
        "Test data",
        transform=axes.transAxes,
        ha="right",
        va="bottom",
        fontsize=6,
    )


if __name__ == "__main__":
    apply_plot_style()

    print("Loading data...")
    prior_a_matrix = pd.read_csv("results/simulations.prior_a.csv", index_col=0)
    prior_corr = pd.read_csv(
        "results/simulations.prior_correlations.csv", dtype={"seq": str}
    ).set_index("seq")

    obs_corr = pd.read_csv(
        "results/simulations.corrs.csv", dtype={"seq": str}
    ).set_index("seq")
    inferred_corr = pd.read_csv(
        "results/simulations.corrs.csv", dtype={"seq": str}
    ).set_index("seq")
    inferred_a_matrix = pd.read_csv(
        "results/simulations.inferred_interaction_strength.csv", index_col=0
    )

    pred = pd.read_csv("results/simulations.pred.csv", index_col=0)
    train = pd.read_csv(
        "data/processed/simulations.train.csv", index_col=0
    ).join(pred, rsuffix="_pred")
    test = pd.read_csv("results/simulations.test_pred.csv", index_col=0)
    r2 = pd.read_csv("results/simulations.r2.csv", index_col=0)

    print("Making figure...")
    fig, subplots = plt.subplots(
        2, 4, figsize=(1.2 * FIG_WIDTH, 0.475 * FIG_WIDTH)
    )

    print("  Plotting prior a matrix...")
    axes = subplots[0, 0]
    plot_interaction_matrix(
        prior_a_matrix,
        axes,
        cbar_label="True interaction strength\n" + r"($1/a_{ij}\times 10^{3}$)",
        vmax=25000,
        scale_factor=1e-3,
    )

    print("  Plotting prior correlation landscape...")
    axes = subplots[0, 1]
    plot_correlation_U_sites(prior_corr, axes, y="cor")
    axes.set(ylabel="True prior correlation")

    print("  Plotting observed correlation landscape...")
    axes = subplots[0, 2]
    plot_correlation_U_sites(obs_corr, axes, y="emp_cor")
    axes.set(ylabel="Observed correlation")

    print("  Plotting prior vs inferred correlation landscape...")
    axes = subplots[0, 3]
    plot_prior_vs_inferred_corr(prior_corr, inferred_corr, axes)

    axes = subplots[1, 0]
    plot_interaction_matrix(
        inferred_a_matrix,
        axes,
        cbar_label="Inferred interaction strength\n"
        + r"($1/a_{ij}\times 10^{-3}$)",
        vmax=25000,
        scale_factor=1e-3,
    )

    print("  Plotting predictions vs true in training sequences")
    axes = subplots[1, 1]
    plot_train_pred_comparison(
        train, axes, lims=(-8, 8), x="f_pred", y="f", cmap="magma"
    )
    axes.set(ylabel="True fitness", xlabel="Predicted fitness")

    print("  Plotting predictions vs true in held-out sequences")
    axes = subplots[1, 2]
    plot_test_pred_comparison(test, axes, lims=(-8, 8))
    coverage = np.mean(
        (test["f"] > test["ci_95_lower"]) & (test["f"] < test["ci_95_upper"])
    )
    print(f"  Coverage of 95% CI: {coverage * 100:.2f}")

    print("  Plotting R2 vs training set size for model comparison...")
    axes = subplots[1, 3]
    plot_cv_r2_curves(r2, axes)

    print("  Saving figure...")
    fig.tight_layout()
    fig.subplots_adjust(
        left=0.05, right=0.98, top=0.98, bottom=0.125, hspace=0.35, wspace=0.75
    )
    add_panel_labels(
        subplots, ["A", "B", "C", "D", "E", "F", "G", "H"], x_offset=-0.22
    )
    fig.savefig("figures/figure2.png", dpi=300)
    fig.savefig("figures/figure2.svg", dpi=300)
    print("Done.")
