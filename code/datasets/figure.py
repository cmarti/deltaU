import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from code.plot_utils import (
    plot_correlation_landscape,
    plot_interaction_matrix,
    plot_pred_vs_obs_corr,
    apply_plot_style,
    add_panel_labels,
    FIG_WIDTH,
    MODELS_PALETTE,
    LINEPLOT_KWARGS,
)


def plot_cv_r2_curves(r2, axes):
    for model, color in MODELS_PALETTE.items():
        df = r2.loc[r2["model"] == model, :]
        sns.lineplot(
            x="p",
            y="r2_test",
            data=df,
            label=model,
            ax=axes,
            color=color,
            **LINEPLOT_KWARGS,
        )
    axes.set(
        xlabel="Proportion of training data",
        ylabel=r"Test $R^2$",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axes.legend(loc=4)


if __name__ == "__main__":
    dataset_names = ["smn1", "dmsc"]
    apply_plot_style()

    print("Making figure...")
    fig, subplots = plt.subplots(
        2, 3, figsize=(0.85 * FIG_WIDTH, 0.475 * FIG_WIDTH)
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
        print(inferred_corr.loc[inferred_corr['d'] == 1,:])
        # r2 = pd.read_csv(f"results/{dataset_name}.r2.csv")

        print("  Plotting empirical correlation landscape...")
        axes = ax_row[0]
        plot_correlation_landscape(inferred_corr, axes, y="emp_cor")
        axes.set(ylabel="Observed correlation")
        
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
        )
        # print("  Plotting R2 vs training set size for model comparison...")
        # axes = subplots[3]
        # plot_cv_r2_curves(r2, axes)

    print("  Saving figure...")
    fig.tight_layout()
    fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.95)
    add_panel_labels(subplots, ["A", "B", "C", "D", "E", "F"], x_offset=-0.22)
    fig.savefig("figures/datasets_figure.png", dpi=300)
    fig.savefig("figures/datasets_figure.svg", dpi=300)
    print("Done.")
