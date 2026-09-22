from code.plot_utils import (
    FIG_WIDTH,
    POSITION_LABELS,
    add_panel_labels,
    apply_plot_style,
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_gamma_i_to_j(
    gamma_i_to_j, position_labels, axes, values="gamma", cmap="Blues_r"
):

    if values == "gamma":
        gamma_label = r"$\gamma_{i \to j}$"
    elif values == "correlation":
        gamma_label = r"$\widetilde\gamma_{i \to j}$"
    else:
        raise ValueError(f"Invalid values argument: {values}")

    m = pd.pivot_table(
        gamma_i_to_j, index="site_i", columns="site_j", values=values
    )
    sns.heatmap(
        m,
        ax=axes,
        cmap=cmap,
        vmin=0,
        vmax=1,
        square=True,
        cbar_kws={"label": gamma_label, "shrink": 0.7},
    )
    axes.set(
        xticks=np.arange(len(position_labels)) + 0.5,
        yticks=np.arange(len(position_labels)) + 0.5,
        xticklabels=position_labels,
        yticklabels=position_labels,
        xlabel="Site $j$",
        ylabel="Site $i$",
    )


def plot_gamma_D_pairs(
    gamma_UD, position_labels, axes, D, values="gamma_UD", cmap="Blues_r"
):
    if values == "gamma_UD":
        gamma_label = r"$\gamma_{\{i,j\}}$"
    elif values == "cor_UD":
        gamma_label = r"$\widetilde\gamma_{\{i,j\}}$"
    else:
        raise ValueError(f"Invalid values argument: {values}")

    D_label = ",".join([str(i) for i in D])
    S_not_D = [i for i in position_labels if i not in D]
    gamma_U_D = gamma_UD.loc[gamma_UD["D"].astype(str) == D_label, :].copy()
    gamma_U_D["i"] = [int(d.split(",")[0]) for d in gamma_U_D["U"]]
    gamma_U_D["j"] = [int(d.split(",")[1]) for d in gamma_U_D["U"]]
    m = (
        pd.pivot_table(gamma_U_D, index="i", columns="j", values=values)
        .reindex(S_not_D)
        .T.reindex(S_not_D)
        .T.fillna(0)
    )
    m = m + m.T
    np.fill_diagonal(m.values, np.nan)

    sns.heatmap(
        m,
        ax=axes,
        cmap=cmap,
        vmin=0,
        vmax=1,
        square=True,
        cbar_kws={
            "label": gamma_label + "({" + D_label + "})",
            "shrink": 0.8,
        },
    )
    axes.set(
        ylabel="Site $i$",
        xlabel="Site $j$",
    )


if __name__ == "__main__":
    dataset_name = "intron.30C"
    position_labels = POSITION_LABELS[dataset_name]
    model_label = "ssVC"
    apply_plot_style()

    print(f"Plotting {model_label} model fit for {dataset_name} dataset")

    ##########################################################################

    print("Loading data for plotting")
    fpath = f"results/{dataset_name}.{model_label}.gamma_i_to_j.csv"
    gamma_i_to_j = pd.read_csv(fpath, index_col=0)

    fpath = f"results/{dataset_name}.{model_label}.gamma_i_to_jk.csv"
    gamma_i_to_jk = pd.read_csv(fpath, index_col=0)
    print(gamma_i_to_jk)

    print("Loading gamma_UD statistics for pairs of sites")
    fpath = f"results/{dataset_name}.{model_label}.gamma_UD_pairs.csv"
    gamma_UDs = pd.read_csv(fpath, index_col=0)

    ##########################################################################

    print("Making figure...")
    fig, subplots = plt.subplots(
        2, 4, figsize=(1.12 * FIG_WIDTH, FIG_WIDTH * 0.425)
    )

    axes = subplots[0, 0]
    plot_gamma_i_to_j(gamma_i_to_j, position_labels, axes)

    axes = subplots[1, 0]
    plot_gamma_i_to_j(
        gamma_i_to_j,
        position_labels,
        axes,
        values="correlation",
    )

    axes = subplots[0, 1]
    plot_gamma_D_pairs(
        gamma_i_to_jk, position_labels, axes, D=[5], values="gamma_UD"
    )

    axes = subplots[1, 1]
    plot_gamma_D_pairs(
        gamma_i_to_jk, position_labels, axes, D=[5], values="cor_UD"
    )

    axes = subplots[0, 2]
    plot_gamma_D_pairs(
        gamma_i_to_jk, position_labels, axes, D=[21], values="gamma_UD"
    )

    axes = subplots[1, 2]
    plot_gamma_D_pairs(
        gamma_i_to_jk, position_labels, axes, D=[21], values="cor_UD"
    )

    axes = subplots[0, 3]
    plot_gamma_D_pairs(
        gamma_UDs, position_labels, axes, D=[2, 21], values="gamma_UD"
    )

    axes = subplots[1, 3]
    plot_gamma_D_pairs(
        gamma_UDs, position_labels, axes, D=[2, 21], values="cor_UD"
    )

    sns.despine(top=False, right=False)

    print("  Saving figure...")
    fig.subplots_adjust(
        wspace=0.5, hspace=0.4, left=0.05, right=0.95, bottom=0.125, top=0.95
    )
    add_panel_labels(
        subplots.flatten(),
        labels=["A", "B", "C", "D", "E", "F", "G", "H"],
        x_offset=-0.22,
        y_offset=1.075,
    )
    fig.savefig("figures/gamma_statistics.png", dpi=300)
    fig.savefig("figures/gamma_statistics.svg", dpi=300)

    print("Done.")
