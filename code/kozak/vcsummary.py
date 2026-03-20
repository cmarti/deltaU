import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs

from scipy.stats import pearsonr
from code.figures.plot_utils import FIG_WIDTH


if __name__ == "__main__":
    dataset = "kozak"
    sl = 6
    upstream_bases = 20
    distance_start_codon = 4
    seq_length = 9
    pos_labels = [str(x) for x in range(-sl - 1, -1)]
    sd_start = upstream_bases - distance_start_codon - seq_length
    x1, x2 = 0.015, 0.5
    y1, y2, y3 = 0.965, 0.64, 0.34

    print("Loading data")
    fpath = "results/vcregression.empirical_distance_correlation.csv"
    dcor = pd.read_csv(fpath, index_col=0)

    fpath = "results/vcregression.variance_components.csv"
    vc = pd.read_csv(fpath, index_col=0).drop(0, axis=0)

    fpath = "results/vcregression.map.variance_components.csv"
    map_vc = pd.read_csv(fpath, index_col=0)

    train = pd.read_csv(f"processed/{dataset}.train.csv", index_col=0)
    test = pd.read_csv(f"processed/{dataset}.test.csv", index_col=0)

    map = pd.read_csv("results/vcregression.map.csv", index_col=0)
    pred = pd.read_csv("results/vcregression.test_pred.csv", index_col=0)

    test = test.join(pred, rsuffix="_pred")
    train = train.join(map, rsuffix="_pred")

    fpath = "results/vcregression.map.site_marginal_epistasis.csv"
    marginal_sites = pd.read_csv(fpath, index_col=0)

    fpath = "results/vcregression.map.pairwise_marginal_epistasis.csv"
    marginal_pw = pd.read_csv(fpath, index_col=0)

    fig = plt.figure(figsize=(FIG_WIDTH * 0.55, 0.6 * FIG_WIDTH))
    gs = gs.GridSpec(3, 4, width_ratios=[1, 0.05, 1, 0.05])

    print("Plotting empirical distance correlation")
    axes = fig.add_subplot(gs[0, 0])
    axes.scatter(dcor["d"], dcor["rho"], color="black", s=5)
    axes.plot(dcor["d"], dcor["rho"], color="black", lw=1)
    axes.set(
        ylabel="Empirical correlation",
        ylim=(None, 1.05),
        xlabel="Hamming distance",
        xticks=np.arange(dcor.shape[0]),
        yticks=np.linspace(0, 1, sl),
    )
    axes.axhline(0, linestyle="--", color="grey", lw=0.75)
    axes.text(
        x1, y1, "A", fontsize=10, weight="bold", transform=fig.transFigure
    )

    print("Plotting prior variance components")
    axes = fig.add_subplot(gs[0, 2])
    axes.bar(x=vc["k"], height=100 * vc["var_perc"], color="black")
    axes.set(
        xlabel="Interaction order $k$",
        ylabel="% variance explained",
        xticks=np.arange(1, sl + 1),
        ylim=(0, None),
    )

    axes = axes.twinx()
    axes.scatter(vc["k"], 100 * vc["var_perc_cum"], color="grey", s=5)
    axes.plot(vc["k"], 100 * vc["var_perc_cum"], color="grey", lw=1)
    axes.tick_params(axis="y", colors="grey")
    axes.spines["right"].set_color("grey")
    axes.set(ylim=(0, 105))
    axes.set_ylabel("% cumulative variance", color="grey")
    axes.text(
        x2, y1, "B", fontsize=10, weight="bold", transform=fig.transFigure
    )

    print("Plotting predicted vs. observed in training set")
    axes = fig.add_subplot(gs[1, 0])
    cbar_axes1 = fig.add_subplot(gs[1, 1])
    lims = (map["f"].min() - 0.25, map["f"].max() + 0.25)
    bins = np.linspace(lims[0], lims[-1], 100)
    x, y = train["f"], train["y"]
    r2 = pearsonr(x, y)[0] ** 2
    sns.histplot(
        x=x,
        y=y,
        cmap="inferno",
        ax=axes,
        bins=(bins, bins),
        cbar_ax=cbar_axes1,
        cbar=True,
        cbar_kws={"label": "# sequences ($10^{3}$)", "shrink": 0.95},
    )
    axes.set(
        xlim=lims,
        ylim=lims,
        # xticks=np.arange(2),
        xlabel="Predicted log(GFP)",
        ylabel="Measured log(GFP)",
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
    axes.text(
        0.05,
        0.95,
        "$R^2$" + "={:.2f}".format(r2),
        transform=axes.transAxes,
        va="top",
        ha="left",
        fontsize=6,
    )
    axes.text(
        x1, y2, "C", fontsize=10, weight="bold", transform=fig.transFigure
    )

    print("Plotting predicted vs. observed in test set")
    axes = fig.add_subplot(gs[1, 2])
    x, xerr = test["f"], 2 * test["f_std"]
    y, yerr = test["y"], 2 * np.sqrt(test["y_var"])
    r2 = pearsonr(y, x)[0] ** 2

    axes.errorbar(
        x,
        y,
        xerr=xerr,
        yerr=yerr,
        color="black",
        fmt="o",
        alpha=0.2,
        markeredgewidth=0,
        markersize=2.5,
        lw=0.5,
    )
    axes.set(
        xlim=lims,
        ylim=lims,
        # xticks=np.arange(4),
        xlabel="Predicted log(GFP)",
        ylabel="Measured log(GFP)",
    )
    axes.axline((0, 0), (1, 1), lw=0.5, c="grey", linestyle="--")
    axes.text(
        0.05,
        0.95,
        "$R^2$" + "={:.2f}\nn={}".format(r2, test.shape[0]),
        transform=axes.transAxes,
        va="top",
        ha="left",
        fontsize=6,
    )
    axes.text(
        0.95,
        0.05,
        "Test data",
        transform=axes.transAxes,
        ha="right",
        va="bottom",
        fontsize=6,
    )
    axes.text(
        x2, y2, "D", fontsize=10, weight="bold", transform=fig.transFigure
    )

    print("Plotting site marginal variance components in the MAP")
    axes = fig.add_subplot(gs[2, 0])
    cbar_axes2 = fig.add_subplot(gs[2, 1])
    marginal_sites.index = pos_labels
    label = "% variance explained by\n interactions involving site $i$"
    sns.heatmap(
        marginal_sites.T.iloc[::-1, :] * 100,
        ax=axes,
        cmap="Greys",
        cbar_ax=cbar_axes2,
        cbar_kws={"label": label, "shrink": 0.8},
        vmin=0,
        # vmax=20,
    )
    sns.despine(ax=axes, top=False, right=False)
    axes.set(
        ylabel="Order of interaction $k$", xlabel="Position $i$ relative to AUG"
    )
    axes.set_xticklabels(axes.get_xticklabels(), rotation=90)
    axes.text(
        x1, y3, "E", fontsize=10, weight="bold", transform=fig.transFigure
    )

    print(
        "Plotting marginal variance components between pairs of sites in the MAP"
    )
    axes = fig.add_subplot(gs[2, 2])
    cbar_axes3 = fig.add_subplot(gs[2, 3])
    label = "% pairwise and higher order\nvariance explained by\n interactions involving $i,j$"
    marginal_pw["high_order"] = marginal_pw["sum"] - marginal_pw["2"]
    df1 = (
        pd.pivot_table(marginal_pw, index="i", columns="j", values="2").fillna(
            0
        )
        / map_vc["vc"].loc[2]
        * 100
    )
    df2 = (
        pd.pivot_table(
            marginal_pw, index="i", columns="j", values="high_order"
        ).fillna(0)
        / map_vc["vc"][1:].sum()
        * 100
    )

    m = np.zeros((sl, sl))
    m[:-1, 1:] = df2.values
    m[1:, :-1] += df1.values.T
    df = pd.DataFrame(m, index=pos_labels, columns=pos_labels)

    sns.heatmap(
        df,
        ax=axes,
        cmap="Greys",
        cbar_kws={"label": label, "shrink": 0.8},
        vmin=0,
        # vmax=20,
        cbar_ax=cbar_axes3,
    )
    sns.despine(ax=axes, top=False, right=False)
    axes.set(
        xlabel="Position $i$ relative to AUG",
        ylabel="Position $j$ relative to AUG",
    )
    axes.text(
        x2, y3, "F", fontsize=10, weight="bold", transform=fig.transFigure
    )
    axes.set_xticklabels(axes.get_xticklabels(), rotation=90)

    print("Adjusting panel positions")
    fig.subplots_adjust(
        bottom=0.1, left=0.125, hspace=0.5, wspace=1.0, top=0.95, right=0.975
    )
    for cbar_axes in [cbar_axes1, cbar_axes2, cbar_axes3]:
        pos = cbar_axes.get_position()
        pos.p0[0] -= 0.105
        pos.p1[0] -= 0.105
        cbar_axes.set_position(pos)
        sns.despine(ax=cbar_axes, top=False, right=False)

    cbar_axes1.set_yticks([0, 5, 10, 15, 20])
    fig.savefig("figures/figure2.png", dpi=300)
    fig.savefig("figures/figure2.svg", dpi=300)
