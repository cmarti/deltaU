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
from itertools import combinations


def plot_gamma_i_to_j(
    gamma_i_to_j, position_labels, axes, values="gamma", cmap="Blues_r"
):

    if values == "gamma":
        gamma_label = r"$\gamma_{(A_i,B_i) \to (A_j,B_j)}$"
    elif values == "correlation":
        gamma_label = r"$\widetilde\gamma_{(A_i,B_i) \to (A_j,B_j)}$"
    else:
        raise ValueError(f"Invalid values argument: {values}")

    m = pd.pivot_table(
        gamma_i_to_j, index="mut_i", columns="mut_j", values=values
    )
    mutations = [
        ("A", "G"),
        ("G", "U"),
        ("C", "G"),
        ("A", "C"),
        ("A", "U"),
        ("C", "U"),
    ]
    idx = []
    for p in position_labels:
        for a1, a2 in mutations:
            idx.append(f"{a1}{p}{a2}")
    m = m.reindex(index=idx, columns=idx)

    sns.heatmap(
        m,
        ax=axes,
        cmap=cmap,
        vmin=0,
        vmax=1,
        square=True,
        cbar_kws={"label": gamma_label, "shrink": 0.7},
        rasterized=True,
    )
    axes.set(
        xticks=6 * np.arange(len(position_labels)) + 3,
        yticks=6 * np.arange(len(position_labels)) + 3,
        xticklabels=position_labels,
        yticklabels=position_labels,
        xlabel="Site $j$",
        ylabel="Site $i$",
    )
    for i in range(8):
        axes.axhline(i * 6, color="black", lw=0.5)
        axes.axvline(i * 6, color="black", lw=0.5)


def plot_gamma_D_pairs(
    gamma_UD, position_labels, axes, D, D_label, values="gamma", cmap="Blues_r"
):
    if values == "gamma":
        gamma_label = r"$\gamma_{\{(A_i, B_i),(A_j, B_j)\}}$"
    elif values == "correlation":
        gamma_label = r"$\widetilde\gamma_{\{(A_i, B_i),(A_j, B_j)\}}$"
    else:
        raise ValueError(f"Invalid values argument: {values}")

    df = gamma_UD.loc[gamma_UD["D"] == D, :]
    position_labels = [
        p
        for p in position_labels
        if (p in df["site_i"].values or p in df["site_j"].values)
    ]
    m = pd.pivot_table(df, index="mut_i", columns="mut_j", values=values)
    mutations = [
        ("A", "G"),
        ("G", "U"),
        ("C", "G"),
        ("A", "U"),
        ("A", "C"),
        ("C", "U"),
    ]
    idx = []
    for p in position_labels:
        for a1, a2 in mutations:
            label = f"{a1}{p}{a2}"
            idx.append(label)

    m = m.reindex(index=idx, columns=idx).fillna(0)
    m = m + m.T
    m[m == 0.0] = np.nan

    D_label = D.replace("-", ",")
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
        rasterized=True,
    )
    axes.set(
        xticks=6 * np.arange(len(position_labels)) + 3,
        yticks=6 * np.arange(len(position_labels)) + 3,
        xticklabels=position_labels,
        yticklabels=position_labels,
        ylabel="Site $i$",
        xlabel="Site $j$",
    )
    for i in range(len(position_labels)):
        axes.axhline(i * 6, color="black", lw=0.5)
        axes.axvline(i * 6, color="black", lw=0.5)


if __name__ == "__main__":
    dataset_name = "intron.30C"
    position_labels = POSITION_LABELS[dataset_name]
    model_label = "ssVC"
    apply_plot_style()

    print(f"Plotting {model_label} model fit for {dataset_name} dataset")

    ##########################################################################

    print("Loading data for plotting")
    fpath = f"results/{dataset_name}.{model_label}.gamma_AiBi_to_AjBj.csv"
    gamma_AiBi_to_AjBj = pd.read_csv(fpath, index_col=0)
    print(gamma_AiBi_to_AjBj)

    fpath = f"results/{dataset_name}.{model_label}.gamma_AiBiAjBj_D.csv"
    gamma_AiBiAjBj_D = pd.read_csv(fpath, index_col=0)
    print(gamma_AiBiAjBj_D)

    ##########################################################################

    print("Making figure...")
    fig, subplots = plt.subplots(
        2, 4, figsize=(1.12 * FIG_WIDTH, FIG_WIDTH * 0.425)
    )

    axes = subplots[0, 0]
    plot_gamma_i_to_j(gamma_AiBi_to_AjBj, position_labels, axes)

    axes = subplots[1, 0]
    plot_gamma_i_to_j(
        gamma_AiBi_to_AjBj,
        position_labels,
        axes,
        values="correlation",
    )

    D = "C21U"
    D_label = r"$(U_{21}, C_{21})$"
    axes = subplots[0, 1]
    plot_gamma_D_pairs(
        gamma_AiBiAjBj_D,
        position_labels,
        axes,
        D=D,
        D_label=D_label,
        values="gamma",
    )

    axes = subplots[1, 1]
    plot_gamma_D_pairs(
        gamma_AiBiAjBj_D,
        position_labels,
        axes,
        D=D,
        D_label=D_label,
        values="correlation",
    )

    D = "G21C"
    D_label = r"$(G_{21}, C_{21})$"
    axes = subplots[0, 2]
    plot_gamma_D_pairs(
        gamma_AiBiAjBj_D,
        position_labels,
        axes,
        D=D,
        D_label=D_label,
        values="gamma",
    )

    axes = subplots[1, 2]
    plot_gamma_D_pairs(
        gamma_AiBiAjBj_D,
        position_labels,
        axes,
        D=D,
        D_label=D_label,
        values="correlation",
    )

    D = "C2G-G21C"
    D_label = r"$(C_2,G_2), (G_{21}, C_{21})$"
    axes = subplots[0, 3]
    plot_gamma_D_pairs(
        gamma_AiBiAjBj_D,
        position_labels,
        axes,
        D=D,
        D_label=D_label,
        values="gamma",
    )

    axes = subplots[1, 3]
    plot_gamma_D_pairs(
        gamma_AiBiAjBj_D,
        position_labels,
        axes,
        D=D,
        D_label=D_label,
        values="correlation",
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
    fig.savefig("figures/gamma_allele_statistics.png", dpi=300)
    fig.savefig("figures/gamma_allele_statistics.svg", dpi=300)

    print("Done.")
