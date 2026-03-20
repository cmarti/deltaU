import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import pearsonr
import gpmap.plot.mpl as mplot
from gpmap.space import SequenceSpace

from code.plot_utils import (
    plot_correlation_landscape,
    apply_plot_style,
    FIG_WIDTH,
    MODELS_PALETTE,
    LINEPLOT_KWARGS,
)



if __name__ == "__main__":
    apply_plot_style()
    dataset_label = 'intron.30C'
    print(f"Plotting model fit for {dataset_label} dataset")
    
    print('  Model predictions and test data...')
    pred = pd.read_csv(f"results/{dataset_label}.ler.landscape.csv", index_col=0)
    test = pd.read_csv(f"data/processed/{dataset_label}.test.csv", index_col=0)
    test = test.join(pred)

    print('  Plotting predicted vs observed fitness in test data...')
    fig, axes = plt.subplots(1, 1, figsize=(0.33 * FIG_WIDTH, 0.33 * FIG_WIDTH))
    axes.scatter(test["f"], test["y"], s=5, lw=0, c="black", alpha=0.5)
    r2 = pearsonr(test["f"], test["y"])[0] ** 2
    axes.text(
        0.05,
        0.95,
        r"$R^2$" + f"={r2:.2f}",
        transform=axes.transAxes,
        ha="left",
        va="top",
        fontsize=8,
    )
    lims = (-9, 4)
    axes.set(
        xlabel="Predicted fitness",
        ylabel="Measured fitness",
        xlim=lims,
        ylim=lims,
        aspect="equal",
    )
    axes.axline(
        (0, 0), slope=1, color="grey", linestyle="--", lw=0.75, alpha=0.5
    )
    # axes.axhline(0, color="grey", linestyle="--", lw=0.75, alpha=0.5)
    # axes.axvline(0, color="grey", linestyle="--", lw=0.75, alpha=0.5)

    print("  Saving figure...")
    fig.tight_layout()
    fig.savefig(f"figures/{dataset_label}.ler.pred.png", dpi=300)
    print("Done.")
