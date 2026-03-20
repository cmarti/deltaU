import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import pearsonr
import gpmap.plot.mpl as mplot
from gpmap.space import SequenceSpace
from code.plot_utils import (
    apply_plot_style,
    FIG_WIDTH,
    MODELS_PALETTE,
    LINEPLOT_KWARGS,
)


def plot_interaction_matrix(a_matrix, axes):
    im = axes.imshow(a_matrix, cmap="binary")
    labels = np.arange(1, 9)
    axes.set(
        xlabel="Site 1",
        ylabel="Site 2",
        xticks=np.arange(8),
        yticks=np.arange(8),
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.colorbar(
        im,
        ax=axes,
        fraction=0.046,
        pad=0.04,
        label="Interaction strength ($1/a_{ij}$)",
    )


if __name__ == "__main__":
    dataset_name = "dmsc"
    apply_plot_style()

    print("Loading data...")
    a_matrix = pd.read_csv(
        f"results/{dataset_name}.inferred_interaction_strength.csv", index_col=0
    )

    print("Making figure...")
    fig, axes = plt.subplots(
        1, 1, figsize=(0.33 * FIG_WIDTH, 0.5 * 0.55 * FIG_WIDTH)
    )

    print("  Plotting inferred a matrix...")
    plot_interaction_matrix(a_matrix, axes)

    print("  Saving figure...")
    fig.tight_layout()
    fig.subplots_adjust(left=0.15, bottom=0.15, top=0.95)
    fig.savefig(f"figures/{dataset_name}_inferred_a_matrix.png", dpi=300)
    print("Done.")
