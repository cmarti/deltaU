from code.plot_utils import POSITION_LABELS, apply_plot_style
from itertools import combinations, product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __name__ == "__main__":
    dataset_name = "intron.30C"

    positions_labels = POSITION_LABELS[dataset_name]
    positions = np.arange(len(positions_labels))
    apply_plot_style()

    mut_effs = pd.read_csv(f"results/{dataset_name}.mut_effs.csv", index_col=0)
    epistatic_coeffs = pd.read_csv(
        f"results/{dataset_name}.epistatic_coeffs.csv", index_col=0
    )

    mutations = list(combinations("ACGU", 2))

    for U in [(0, 7), (3, 4)]:
        print("  Plotting epistatic coefficients for positions", U)

        fig, subplots = plt.subplots(
            len(mutations),
            len(mutations),
            figsize=(6, 6),
        )

        xmin, xmax = -8.5, 8.5
        for i, j in product(range(len(mutations)), repeat=2):
            axes = subplots[i, j]
            a_i_1, a_i_2 = mutations[i]
            a_j_1, a_j_2 = mutations[j]
            label = f"{a_i_1}{positions_labels[U[0]]}{a_i_2}_{a_j_1}{positions_labels[U[1]]}{a_j_2}"
            print("    Plotting epistatic coefficients for mutations", label)

            if i == j:
                axes.text(
                    0.5,
                    0.5,
                    f"Mutation {a_i_1}>{a_i_2}",
                    ha="center",
                    va="center",
                    fontsize=7,
                )
                axes.axis("off")

            else:
                bins = np.linspace(xmin, xmax, 40)
                values = epistatic_coeffs[label].values
                axes.hist(
                    values,
                    bins=bins,
                    color="grey",
                    alpha=1,
                )
                axes.axvline(
                    x=0, color="grey", linestyle="--", lw=0.5, alpha=0.5
                )

                label = f"{a_i_1}{positions_labels[U[0]]}{a_i_2}-{a_j_1}{positions_labels[U[1]]}{a_j_2}\n"
                label += (
                    r"$\overline{\epsilon_{UC}}$"
                    + f" = {np.mean(values):.2f}\n"
                )
                label += (
                    r"$\operatorname{Var}[\epsilon_{UC}]$"
                    + f" = {np.var(values):.2f}"
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
                axes.set(
                    xlim=(xmin, xmax),
                    ylim=(0, 1000),
                    # xticks=[xmin, 0, xmax],
                    yticks=[],
                    xticklabels=[],
                )

        for axes in subplots[-1, :-1]:
            axes.set(
                xticks=[-5, 0, 5],
                xticklabels=[-5, 0, 5],
                xlabel="Epistatic coefficient\n $\\epsilon_{UC}$",
            )

        for axes in subplots[1:, 0]:
            axes.set(ylabel="Density")

        fig.supylabel(
            f"Mutation at position {positions_labels[U[0]]}",
            fontsize=8,
            x=0.005,
            y=0.5,
            va="center",
        )
        fig.suptitle(
            f"Mutation at position {positions_labels[U[1]]}",
            fontsize=8,
            x=0.525,
            y=0.975,
            ha="center",
        )

        print("  Rendering figure...    ")
        fig.subplots_adjust(
            left=0.06,
            right=0.975,
            bottom=0.08,
            top=0.95,
            wspace=0.1,
            hspace=0.1,
        )
        fig.savefig(
            f"figures/{dataset_name}.ler.epistatic_coeffs.{'_'.join([str(positions_labels[i]) for i in U])}.png",
            dpi=300,
        )

    print("Done.")
