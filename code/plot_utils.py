from itertools import combinations

import gpmap.plot.mpl as mplot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import cm
from scipy.stats import pearsonr

######################
# Plotting constants #
######################
DATASETS = ["smn1", "dmsc", "gb1", "fyn-sh3", "intron.30C"]
FIG_WIDTH = 7.5
STYLE_PATH = "style.sty"
LINEPLOT_LW = 0.75
LINEPLOT_KWARGS = {
    "lw": LINEPLOT_LW,
    "err_style": "bars",
    "err_kws": {
        "capsize": LINEPLOT_LW * 0.2,
        "capthick": 0,
        "lw": LINEPLOT_LW,
        "elinewidth": 0.5,
    },
    # "errorbar": "sd",
    "ci": "sd",
}

GREYS = cm.get_cmap("binary")

MODELS_PALETTE = {
    "MEI": GREYS(0.2),
    "CN": GREYS(0.45),
    "VC": GREYS(0.7),
    "LER": GREYS(0.99),
    "Pairwise": 'grey',
    "Additive": 'grey',
}
MODELS_STYLES = {
    "MEI": "-",
    "CN": "-",
    "VC": "-",
    "LER": "-",
    "Pairwise": "--",
    "Additive": ":",
}
POSITION_LABELS = {
    "smn1": ["-3", "-2", "-1", "+2", "+3", "+4", "+5", "+6"],
    "dmsc": np.arange(-13, -4),
    "intron": [2, 3, 4, 5, 18, 19, 20, 21],
    "intron.30C": [2, 3, 4, 5, 18, 19, 20, 21],
    "intron.37C": [2, 3, 4, 5, 18, 19, 20, 21],
    "gb1": [39, 40, 41, 54],
    "fyn-sh3": [3, 17, 19, 25, 27, 49, 54],
}


def apply_plot_style(overrides=None):
    """Apply project-wide matplotlib style defaults from an mplstyle file."""
    plt.style.use(STYLE_PATH)
    if overrides:
        plt.rcParams.update(overrides)


def add_r2_label(axes, r2, fontsize=6):
    axes.text(
        0.05,
        0.95,
        r"$R^2$" + f"={r2:.2f}",
        transform=axes.transAxes,
        ha="left",
        va="top",
        fontsize=fontsize,
    )


def plot_pred_vs_obs_corr(corr, axes):
    axes.scatter(
        corr["pred_cor"],
        corr["emp_cor"],
        color="black",
        s=5,
        alpha=0.5,
        lw=0,
    )
    r2 = pearsonr(corr["pred_cor"], corr["emp_cor"])[0] ** 2
    add_r2_label(axes, r2)
    axes.set(
        xlabel="Prior correlation",  # yticklabels=[],
        ylabel="Observed correlation",
    )
    axes.axline(
        (0, 0), slope=1, color="grey", linestyle="--", lw=0.75, alpha=0.5
    )
    axes.axhline(0, color="grey", linestyle="--", lw=0.75, alpha=0.5)
    axes.axvline(0, color="grey", linestyle="--", lw=0.75, alpha=0.5)


def add_panel_labels(subplots, labels, x_offset=-0.1, y_offset=1.05):
    for ax, label in zip(subplots.flat, labels):
        ax.text(
            x_offset,
            y_offset,
            label,
            transform=ax.transAxes,
            fontsize=10,
            fontweight="bold",
            va="top",
            ha="right",
        )


def plot_cv_r2_curves(r2, axes):
    for model, color in MODELS_PALETTE.items():
        df = r2.loc[r2["model"] == model, :].copy()
        sns.lineplot(
            x="p",
            y="r2_test",
            data=df,
            label=model,
            ax=axes,
            linestyle=MODELS_STYLES[model],
            color=color,
            **LINEPLOT_KWARGS,
        )
    axes.set(
        xlabel="Proportion of training data",
        ylabel=r"Test $R^2$",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axes.legend(loc=4, ncol=2)


def plot_train_pred_comparison(train, axes, lims, x='f', y='y'):
    bins = np.linspace(lims[0], lims[-1], 100)
    x, y = train[x], train[y]
    sns.histplot(
        x=x,
        y=y,
        cmap="Greys_r",
        ax=axes,
        bins=(bins, bins),
        cbar=True,
        cbar_kws={
            "label": "# sequences",
            "fraction": 0.046,
            "pad": 0.04,
        },
        rasterized=True,
    )
    axes.set(
        xlim=lims,
        ylim=lims,
        xlabel="Predicted fitness",
        ylabel="Measured fitness",
    )
    axes.axline((0, 0), (1, 1), lw=0.5, c="grey", linestyle="--")
    axes.text(
        0.95,
        0.05,
        "Training data",
        transform=axes.transAxes,
        ha="right",
        va="bottom",
        fontsize=6,
    )
    r2 = pearsonr(x, y)[0] ** 2
    add_r2_label(axes, r2)


def plot_test_pred_comparison(test, axes, lims):
    x, xerr = test["f"], 2 * test["f_std"]
    y, yerr = test["y"], 2 * np.sqrt(test["y_var"])
    r2 = pearsonr(y, x)[0] ** 2

    axes.errorbar(
        x,
        y,
        xerr=xerr,
        yerr=yerr,
        color="grey",
        fmt="none",
        alpha=0.2,
        markeredgewidth=0,
        markersize=2.5,
        lw=0.5,
        zorder=1
    )
    
    axes.scatter(
        x,
        y,
        color="black",
        alpha=0.5,
        s=2,
        lw=0,
        zorder=2
    )
    axes.set(
        xlim=lims,
        ylim=lims,
        xlabel="Predicted fitness",
        ylabel="Measured fitness",
    )
    axes.axline((0, 0), (1, 1), lw=0.5, c="grey", linestyle="--")
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


def arrange_axis(
    axes, x, y, ticks, lims, fontsize=8, xpos=0.52, ypos=0.52, ms=5
):
    axes.set(aspect="equal", xlabel="", ylabel="")
    axes.spines["left"].set(position=("data", 0), zorder=0, alpha=0.5)
    axes.spines["bottom"].set(position=("data", 0), zorder=0, alpha=0.5)
    axes.set(xticks=ticks, yticks=ticks, ylim=lims, xlim=lims)
    axes.plot(
        (1),
        (0),
        ls="",
        marker=">",
        ms=ms,
        color="k",
        transform=axes.get_yaxis_transform(),
        clip_on=False,
    )
    axes.plot(
        (0),
        (1),
        ls="",
        marker="^",
        ms=ms,
        color="k",
        transform=axes.get_xaxis_transform(),
        clip_on=False,
    )
    axes.text(
        1.02,
        xpos,
        f"Diffusion axis {x}",
        transform=axes.transAxes,
        fontsize=fontsize,
        ha="right",
        va="bottom",
    )
    axes.text(
        ypos,
        1.01,
        f"Diffusion axis {y}",
        transform=axes.transAxes,
        fontsize=fontsize,
        ha="left",
        va="top",
    )
    sns.despine(ax=axes)


def calc_hamming_distance(s1, s2):
    return np.sum([a1 != a2 for a1, a2 in zip(s1, s2)])


def calc_edges_df(seqs):
    edges = []
    for i, j in combinations(range(len(seqs)), 2):
        s_i, s_j = seqs[i], seqs[j]
        if calc_hamming_distance(s_i, s_j) == 1:
            edges.append((i, j))
    edges_df = pd.DataFrame(edges, columns=["i", "j"])
    return edges_df


def plot_local_landscape(contrasts, seqs, axes, pos1="$_{2}$", pos2="$_{21}$"):
    edges_df = calc_edges_df(seqs)
    nodes_df = pd.DataFrame(
        {
            "x": [calc_hamming_distance(s, seqs[0]) for s in seqs],
            "f": contrasts["estimate"].values,
            "err": 2 * contrasts["std"].values,
        },
        index=seqs,
    )

    axes.errorbar(
        nodes_df["x"],
        nodes_df["f"],
        yerr=nodes_df["err"],
        fmt="none",
        markersize=0.5,
        color="black",
        lw=0,
        ecolor="black",
        elinewidth=0.5,
        capsize=1,
        capthick=0.5,
    )

    mplot.plot_visualization(
        axes,
        nodes_df,
        x="x",
        y="f",
        edges_df=edges_df,
        nodes_color="black",
        nodes_size=0,
        edges_alpha=1,
        edges_width=0.75,
        edges_color="lightgrey",
    )
    for seq, x, y in zip(seqs, nodes_df["x"], nodes_df["f"]):
        axes.text(
            x,
            y + 0.2,
            seq[0] + pos1 + seq[1] + pos2,
            fontsize=5,
            ha="center",
        )
