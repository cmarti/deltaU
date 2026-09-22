from code.plot_utils import (
    FIG_WIDTH,
    apply_plot_style,
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gpmap.plot.mpl import plot_correlation_U_sites
from scipy.stats import pearsonr


def scatter(x, y, log_r2=True):
    axes.scatter(
        x,
        y,
        color="black",
        s=5,
        alpha=0.5,
        lw=0,
    )
    if log_r2:
        r2 = pearsonr(np.log10(x), np.log10(y))[0] ** 2
    else:
        r2 = pearsonr(x, y)[0] ** 2
    axes.text(
        0.05,
        0.95,
        r"$R^2$" + f"={r2:.2f}",
        transform=axes.transAxes,
        ha="left",
        va="top",
        fontsize=8,
    )
    axes.axline(
        (0, 0), slope=1, color="grey", linestyle="--", lw=0.75, alpha=0.5
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
        fontsize=6,
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
    lambda_U_ler = pd.read_csv(
        "results/simulations.ler.lambda_U.csv", dtype={"U": str}
    ).set_index("U")
    lambda_U_ler["d_jittered"] = np.random.normal(
        lambda_U_ler["k"], scale=0.05
    )
    lambda_U_ssVC = pd.read_csv(
        "results/simulations.ssVC.lambda_U.csv", dtype={"U": str}
    ).set_index("U")
    lambda_U_ssVC["d_jittered"] = np.random.normal(
        lambda_U_ssVC["k"], scale=0.05
    )

    ler_lambda_U_ler_inferred = pd.read_csv(
        "results/simulations.ler.inferred_lambda_U.ler.csv", dtype={"U": str}
    ).set_index("U")
    ler_lambda_U_ssVC_inferred = pd.read_csv(
        "results/simulations.ler.inferred_lambda_U.ssVC.csv", dtype={"U": str}
    ).set_index("U")
    ler_lambda_U_ssVC_inferred["lambda_U"] = ler_lambda_U_ssVC_inferred[
        "lambdas"
    ]

    ssVC_lambda_U_ler_inferred = pd.read_csv(
        "results/simulations.ssVC.inferred_lambda_U.ler.csv", dtype={"U": str}
    ).set_index("U")
    ssVC_lambda_U_ssVC_inferred = pd.read_csv(
        "results/simulations.ssVC.inferred_lambda_U.ssVC.csv", dtype={"U": str}
    ).set_index("U")
    ssVC_lambda_U_ssVC_inferred["lambda_U"] = ssVC_lambda_U_ssVC_inferred[
        "lambdas"
    ]

    #########################################################################

    print("Making figure...")
    fig, subplots = plt.subplots(
        2,
        3,
        figsize=(0.7 * FIG_WIDTH, 0.45 * FIG_WIDTH),
    )
    subplots = subplots.T

    ###################################
    
    ylim = (5e-3, 8e3)
    xlim = (8e-7, 5e3)
    axes = subplots[0, 0]
    plot_correlation_U_sites(lambda_U_ler, axes, y="lambda_U")
    axes.set(
        yscale="log",
        xlabel="",
        xticklabels=[],
        ylabel=r"LER true prior $\lambda_U$",
        ylim=ylim,
    )

    axes = subplots[1, 0]
    scatter(ler_lambda_U_ler_inferred["lambda_U"], lambda_U_ler["lambda_U"])
    axes.set(
        yscale="log",
        xscale="log",
        xlabel="",
        xticklabels=[],
        yticklabels=[],
        ylim=ylim,
        xlim=xlim,
    )

    axes = subplots[2, 0]
    scatter(ler_lambda_U_ssVC_inferred["lambda_U"], lambda_U_ler["lambda_U"])
    axes.set(
        yscale="log",
        xscale="log",
        xlabel="",
        xticklabels=[],
        yticklabels=[],
        ylim=ylim,
        xlim=xlim,
    )
    
    ###################################
    
    axes = subplots[0, 1]
    plot_correlation_U_sites(lambda_U_ssVC, axes, y="lambda_U")
    axes.set(
        yscale="log",
        xlabel="Interaction order $k$",
        ylabel=r"ssVC true prior $\lambda_U$",
        ylim=ylim,
    )
    
    axes = subplots[1, 1]
    
    scatter(ssVC_lambda_U_ler_inferred["lambda_U"], lambda_U_ssVC["lambda_U"])
    axes.set(
        yscale="log",
        xscale="log",
        xlabel=r"LER inferred prior $\lambda_U$",
        ylim=ylim,
        yticklabels=[],
        xlim=xlim,
    )

    axes = subplots[2, 1]
    scatter(ssVC_lambda_U_ssVC_inferred["lambda_U"], lambda_U_ssVC["lambda_U"])
    axes.set(
        yscale="log",
        xscale="log",
        xlabel=r"ssVC inferred prior $\lambda_U$",
        ylim=ylim,
        yticklabels=[],
        xlim=xlim,
    )

    print("  Saving figure...")
    fig.tight_layout()
    fig.subplots_adjust(left=0.1, right=0.98, top=0.96, bottom=0.12)
    fig.savefig("figures/figureS1.png", dpi=300)
    fig.savefig("figures/figureS1.svg", dpi=300)
    print("Done.")
