from code.plot_utils import (
    FIG_WIDTH,
    POSITION_LABELS,
    add_panel_labels,
    add_r2_label,
    apply_plot_style,
    plot_cv_r2_curves,
    plot_train_pred_comparison,
)

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr

if __name__ == "__main__":
    dataset_name = "intron.30C"
    position_labels = POSITION_LABELS[dataset_name]
    apply_plot_style()

    print(f"Plotting model fit for {dataset_name} dataset")

    print("  Loading R2 curves data")
    r2 = pd.read_csv(f"results/{dataset_name}.r2_curves.csv", index_col=0)

    print("  Loading models predictions...")
    data = pd.read_csv(
        f"results/{dataset_name}.models_predictions.csv", index_col=0
    )

    print("  Plotting cross-validation curves")
    fig, subplots = plt.subplots(1, 3, figsize=(0.95*FIG_WIDTH, FIG_WIDTH * 0.275))

    axes = subplots[0]
    plot_cv_r2_curves(r2, axes)
    axes.set(ylim=(0.2, 0.8))

    print("  Plotting model comparisons in full landscape")
    axes = subplots[1]
    plot_train_pred_comparison(data, axes, lims=(-8, 6), x="LER", y="VC")
    axes.set(xlabel="LER predictions", ylabel="VC predictions", aspect="equal")
    r2 = pearsonr(data["LER"], data["VC"])[0] ** 2
    add_r2_label(axes, r2)

    print("  Plotting model comparisons in held-out data")
    axes = subplots[2]
    test = pd.read_csv(f"data/processed/{dataset_name}.test.csv", index_col=0)
    data = data.loc[test.index, :]
    axes.scatter(data["LER"], data["VC"], alpha=0.5, s=3, c="black", lw=0)
    axes.axline((0, 0), (1, 1), color="gray", ls="--", lw=0.75)
    axes.set(
        xlabel="LER predictions",
        ylabel="VC predictions",
        aspect="equal",
        xlim=(-4.5, 3.5),
        ylim=(-4.5, 3.5),
    )
    r2 = pearsonr(data["LER"], data["VC"])[0] ** 2
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

    add_panel_labels(subplots, ["A", "B", "C"], x_offset=-0.22)

    print("  Saving figure...")
    fig.tight_layout()
    fig.savefig("figures/figureS4.png", dpi=300)
    fig.savefig("figures/figureS4.svg", dpi=300)
    print("Done.")
