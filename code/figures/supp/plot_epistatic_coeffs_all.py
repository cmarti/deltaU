from code.plot_utils import POSITION_LABELS, apply_plot_style
from itertools import combinations, product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __name__ == "__main__":
    dataset_name = "intron.30C"
    positions_labels = POSITION_LABELS[dataset_name]
    positions = np.arange(len(positions_labels))

    mut_effs = pd.read_csv(f"results/{dataset_name}.mut_effs.csv", index_col=0)
    epistatic_coeffs = pd.read_csv(
        f"results/{dataset_name}.epistatic_coeffs.csv", index_col=0
    )
    apply_plot_style()

    fig, subplots = plt.subplots(
        8,
        8,
        figsize=(8, 8),  # sharex=True, sharey=True
    )

    mutations = list(combinations("ACGU", 2))

    for i in positions:
        axes = subplots[i, i]
        axes.text(
            0.5,
            0.5 - 0.075 * (i != 0),
            f"Position {positions_labels[i]}",
            ha="center",
            va="center",
            fontsize=8,
        )
        axes.axis("off")

        if i > 0:
            axes.text(
                0.5,
                0.875,
                "Squared epistatic\n coefficient $\\epsilon^2$",
                ha="center",
                va="top",
                fontsize=7,
            )
            for x, xtick in zip([0, 0.25, 0.5, 0.75, 1], [0, 5, 10, 15, 20]):
                axes.text(
                    x,
                    1.0,
                    f"{xtick}",
                    ha="center",
                    va="top",
                    fontsize=6,
                )

    for i, j in combinations(positions, 2):
        print("    Between positions", i, j)
        labels = []
        for (a_i_1, a_i_2), (a_j_1, a_j_2) in product(mutations, repeat=2):
            label = f"{a_i_1}{positions_labels[i]}{a_i_2}_{a_j_1}{positions_labels[j]}{a_j_2}"
            labels.append(label)

        axes = subplots[i, j]
        bins = np.linspace(0, 20, 50)
        values = epistatic_coeffs[labels].values.flatten() ** 2
        axes.hist(
            values,
            bins=bins,
            color="grey",
            alpha=1,
        )
        axes.axvline(
            x=np.mean(values),
            color="red",
            linestyle="--",
            linewidth=0.5,
        )
        label = f"U={{{positions_labels[i]}, {positions_labels[j]}}}\n"
        label += r"$\overline{\epsilon^2_U}$" + f" = {np.mean(values):.2f}"
        axes.text(
            0.95,
            0.95,
            label,
            transform=axes.transAxes,
            ha="right",
            va="top",
            fontsize=6,
        )
        axes.set(
            xlim=(0, 20), xticks=[0, 5, 10, 15, 20], yticks=[], xticklabels=[]
        )

        bins = np.linspace(-6, 6, 50)
        axes = subplots[j, i]
        values = epistatic_coeffs[labels].values
        values -= values.mean(0, keepdims=True)
        values = values.flatten()
        axes.hist(
            values,
            bins=bins,
            color="grey",
            alpha=1,
        )
        axes.set(
            xticks=[-5, 0, 5],
            yticks=[],
            xticklabels=[],
            xlim=(-6, 6),
            ylim=(0, 15e3),
        )
        label = f"U={{{positions_labels[i]}, {positions_labels[j]}}}\n"
        label += (
            r"$\operatorname{Var}[\epsilon_U]$" + f" = {np.mean(values**2):.2f}"
        )
        axes.text(
            0.95,
            0.95,
            label,
            transform=axes.transAxes,
            ha="right",
            va="top",
            fontsize=6,
        )

    for axes in subplots[-1, :-1]:
        axes.set(
            xticklabels=[-5, 0, 5],
            xlabel="Mean centered\n epistatic coefficient\n $\\epsilon_{UC}-\\overline{\\epsilon_{UC}}$",
        )

    for axes in subplots[1:, 0]:
        axes.set(ylabel="Density")

    print("  Rendering figure...    ")
    fig.subplots_adjust(
        left=0.025, right=0.975, bottom=0.08, top=0.985, wspace=0.1, hspace=0.1
    )
    fig.savefig(
        f"figures/{dataset_name}.ler.epistatic_coeffs.all.png",
        dpi=300,
    )

    print("Done.")
