from code.plot_utils import (
    FIG_WIDTH,
    POSITION_LABELS,
    add_panel_labels,
    apply_plot_style,
    plot_cv_r2_curves,
    plot_pred_vs_obs_corr,
)

import matplotlib.pyplot as plt
import pandas as pd
from gpmap.plot.mpl import plot_correlation_U_sites, plot_interaction_matrix

if __name__ == "__main__":
    dataset_names = ["gb1", "fyn-sh3"]
    apply_plot_style()

    print("Making figure...")
    fig, subplots = plt.subplots(
        2, 4, figsize=(1.2 *FIG_WIDTH, 0.475 * FIG_WIDTH)
    )

    for dataset_name, ax_row in zip(dataset_names, subplots):
        print(f"Loading data for {dataset_name}...")
        a_matrix = pd.read_csv(
            f"results/{dataset_name}.inferred_interaction_strength.csv",
            index_col=0,
        )
        inferred_corr = pd.read_csv(
            f"results/{dataset_name}.corrs.csv", dtype={"seq": str}
        ).set_index("seq")
        r2 = pd.read_csv(f"results/{dataset_name}.r2_curves.csv")

        print("  Plotting empirical correlation landscape...")
        axes = ax_row[0]
        plot_correlation_U_sites(inferred_corr, axes, y="emp_cor")
        axes.set(ylabel="Observed correlation", xlabel='Hamming distance')
        
        print("  Plotting prior vs inferred correlation landscape...")
        axes = ax_row[1]
        plot_pred_vs_obs_corr(inferred_corr, axes)

        print("  Plotting prior a matrix...")
        axes = ax_row[2]
        plot_interaction_matrix(
            a_matrix,
            axes,
            vmax=None,
            position_labels=POSITION_LABELS[dataset_name],
            cbar_label='Interaction strength ($1/a_{ij}$)'
        )
        print("  Plotting R2 vs training set size for model comparison...")
        axes = ax_row[3]
        plot_cv_r2_curves(r2, axes)
        axes.set(ylim=(0.3, 1.))

    print("  Saving figure...")
    fig.tight_layout()
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.1, top=0.95)
    add_panel_labels(subplots, ["A", "B", "C", "D", "E", "F", "G", "H"],
                     x_offset=-0.22)
    fig.savefig("figures/figureS1.png", dpi=300)
    fig.savefig("figures/figureS1.svg", dpi=300)
    print("Done.")
