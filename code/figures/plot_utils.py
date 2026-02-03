import gpmap.plot.mpl as plot
import numpy as np
import pandas as pd
import seaborn as sns


import matplotlib.pyplot as plt

FIG_WIDTH = 8

# Fonts
plt.rcParams["font.family"] = "Nimbus Sans"
plt.rcParams["axes.titlesize"] = 7
plt.rcParams["axes.labelsize"] = 7
plt.rcParams["xtick.labelsize"] = 6
plt.rcParams["ytick.labelsize"] = 6
plt.rcParams["legend.fontsize"] = 6
plt.rcParams["legend.labelspacing"] = 0.1

# plt.rcParams["axes.titlepad"] = 3

# Linewidths
plt.rcParams["axes.linewidth"] = 0.6
plt.rcParams["xtick.major.width"] = 0.6
plt.rcParams["ytick.major.width"] = 0.6
plt.rcParams["xtick.minor.width"] = 0.35
plt.rcParams["ytick.minor.width"] = 0.35


def annotate_seq(
    axes,
    seq,
    df,
    dx,
    dy,
    ha,
    va,
    x="1",
    y="2",
    fontsize=None,
    arrow_size=0.75,
):
    labels = {
        "AAGGAGCAG": r"A$\bf{AGGAG}$CAG",
        "UUAAGGAGC": r"UUA$\bf{AGGAG}$C",
        "UAAGGAGCA": r"UA$\bf{AGGAG}$CA",
        "AGGAGAAUA": r"$\bf{AGGAG}$AAUA",
        "AGGAGGAGC": r"$\bf{AGGAGGAG}$C",
        "GAGUUUAAU": r"$\bf{GAG}$UUUAAU",
        "GAGGUUCAG": r"$\bf{GAGG}$UUCAG",
        "UAGGAGGUA": r"U$\bf{AGGAGGU}$A",
        "GGAGGUACA": r"$\bf{GGAGGU}$ACA",
        "GGAGUUUAA": r"$\bf{GGAG}$UUUAA",
        "GAGGAGGAU": r"$\bf{GAGGAGG}$AU",
        "GGAGGAGAA": r"$\bf{GGAGGAG}$AA",
        "AAGGAAUAU": r"A$\bf{AGGA}$AUAU",
        "GGAGGAAUA": r"$\bf{GGAGG}$AAUA",
        "CAGGAGGUA": r"C$\bf{AGGAGG}$UA",
        "AGGAGGUAC": r"$\bf{AGGAGG}$UAC",
        "AGGAGGAGG": r"$\bf{AGGAGGAGG}$",
        "UUAAGGAGG": r"UUA$\bf{AGGAGG}$",
        "GGAGGUACC": r"$\bf{GGAGG}$UACC",
        "GGAGGAGGU": r"$\bf{GGAGGAGG}$U",
        "UAAGGAGGU": r"UA$\bf{AGGAGG}$U",
        "UUAGGAGGA": r"UU$\bf{AGGAGG}$A",
        "AAGGAGCUG": r"A$\bf{AGGAG}$CUG",
        "CAAAGGAGG": r"CAA$\bf{AGGAGG}$",
        "GGAGGAGGA": r"$\bf{GGAGGAGG}A$",
        "GGAGGUUUA": r"$\bf{GGAGG}$UUUA",
        "AGGAGGUUA": r"$\bf{AGGAGG}$UUA",
    }

    x, y = df.loc[seq, [x, y]]
    axes.annotate(
        labels.get(seq, seq),
        xy=(x, y),
        xytext=(x + dx, y + dy),
        arrowprops=dict(
            facecolor="black",
            shrink=0.05,
            width=1 * arrow_size,
            headwidth=7 * arrow_size,
            headlength=12 * arrow_size,
        ),
        ha=ha,
        va=va,
        fontsize=fontsize,
    )


def plot_path(
    axes, nodes, x="1", y="2", size=40, lw=1.5, seqs=None, vmin=None, vmax=None
):
    if seqs is None:
        seqs = [
            "AGGAGAAUA",
            "AGGAGGAUA",
            "AGGAGGAGA",
            "AGGAGGAGC",
            "UGGAGGAGC",
            "UUGAGGAGC",
            "UUAAGGAGC",
        ]
    sl = len(seqs)
    edf = pd.DataFrame({"i": np.arange(sl - 1), "j": np.arange(1, sl)})
    ndf = nodes.loc[seqs, :]
    v = nodes["function"]
    vmax = v.max()
    plot.plot_nodes(
        axes,
        ndf,
        lw=lw,
        size=size,
        cbar=False,
        vmin=vmin,
        vmax=vmax,
        color="function",
        zorder=4,
        x=x,
        y=y,
    )
    plot.plot_edges(
        axes, ndf, edf, color="black", alpha=1, width=lw, zorder=3, x=x, y=y
    )


def plot_landscape(
    axes,
    ndf,
    edf=None,
    x="1",
    y="2",
    cmap_label="",
    vmin=np.log(1e-7),
    vmax=np.log(1e-3),
    c="function",
    z="3",
    lims=(-3.1, 3.1),
    size=5,
    log_p=True,
    ypos=0.52,
    xpos=0.52,
    label_size=8,
):
    nodes_hist_axes = axes.inset_axes((0.05, 0.875, 0.3, 0.1))
    nodes_cbar_axes = axes.inset_axes((0.05, 0.85, 0.3, 0.02))

    plot.plot_nodes(
        axes,
        ndf,
        x=x,
        y=y,
        sort_by=z,
        sort_ascending=True,
        color=c,
        size=size,
        cmap="viridis",
        cbar_axes=nodes_cbar_axes,
        cbar_label=cmap_label,
        cbar_orientation="horizontal",
        vmin=vmin,
        vmax=vmax,
        rasterized=True,
    )

    plot_function_hist(ndf, vmin, vmax, nodes_hist_axes, c)
    nodes_hist_axes.set_facecolor("none")
    if log_p:
        arrange_cbar(nodes_cbar_axes)
        ticks = np.array(
            [-3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3]
        )
    else:
        ticks = np.array(
            [-2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5]
        )
    arrange_axis(
        axes,
        x,
        y,
        ticks,
        lims,
        ypos=ypos,
        xpos=xpos,
        fontsize=label_size,
    )
    return nodes_cbar_axes, nodes_hist_axes


def plot_function_hist(ndf, vmin, vmax, nodes_hist_axes, c):
    bins = np.linspace(vmin, vmax, 30)
    plot.plot_color_hist(nodes_hist_axes, ndf[c], cmap="viridis", bins=bins)
    nodes_hist_axes.set_ylabel("Frequency", fontsize=7)


def arrange_cbar(nodes_cbar_axes):
    yticklabels = 10 ** np.arange(-7, -2).astype(float)
    nodes_cbar_axes.set_xlabel("Sequence probability", fontsize=7)
    nodes_cbar_axes.set_xticks(np.log(yticklabels))
    nodes_cbar_axes.set_xticklabels(
        ["$10^{-7}$", "$10^{-6}$", "$10^{-5}$", "$10^{-4}$", "$10^{-3}$"],
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
        1.1,
        xpos,
        "Diffusion axis {}".format(x),
        transform=axes.transAxes,
        fontsize=fontsize,
        ha="right",
        va="bottom",
    )
    axes.text(
        ypos,
        1,
        "Diffusion axis {}".format(y),
        transform=axes.transAxes,
        fontsize=fontsize,
        ha="left",
        va="top",
    )
    sns.despine(ax=axes)


def plot_relaxation_times(relaxation_times, axes):
    axes.scatter(
        relaxation_times["k"],
        relaxation_times["relaxation_time"],
        c="black",
        s=5,
    )
    axes.plot(
        relaxation_times["k"],
        relaxation_times["relaxation_time"],
        c="black",
        lw=1,
    )
    axes.axhline(0.25, c="grey", lw=0.75, label="Neutral", linestyle="--")
    axes.legend(loc=1)
    axes.set(
        ylim=(0.2, None),
        # xticks=relaxation_times["k"],
        xticks=[1, 5, 10, 15, 20],
    )
    axes.set_ylabel("Relaxation time\n(expected substitutions)")
    axes.set_xlabel("Diffusion axis")


def add_vcregression_labels(
    axes, nodes_df, fontsize=6, label_path=True, arrow_size=1
):
    annotate_seq(
        axes,
        "UAGGAGGUA",
        nodes_df,
        dx=0.7,
        dy=0.7,
        ha="left",
        va="bottom",
        fontsize=fontsize,
        arrow_size=arrow_size,
    )
    annotate_seq(
        axes,
        "UAAGGAGCA",
        nodes_df,
        dx=-0.7,
        dy=-0.7,
        ha="center",
        va="top",
        fontsize=fontsize,
        arrow_size=arrow_size,
    )
    annotate_seq(
        axes,
        "UUAAGGAGC",
        nodes_df,
        dx=0.7,
        dy=-0.7,
        ha="center",
        va="top",
        fontsize=fontsize,
        arrow_size=arrow_size,
    )
    if label_path:
        annotate_seq(
            axes,
            "AGGAGAAUA",
            nodes_df,
            dx=0.2,
            dy=-1.2,
            ha="left",
            va="bottom",
            fontsize=fontsize,
            arrow_size=arrow_size,
        )
        annotate_seq(
            axes,
            "AGGAGGAGC",
            nodes_df,
            dx=0.1,
            dy=-0.8,
            ha="left",
            va="top",
            fontsize=fontsize,
            arrow_size=arrow_size,
        )


def plot_posterior(
    axes, df, x="x", y="estimate", std="std", color="black", label=None
):
    kwargs = {
        "fmt": "o",
        "lw": 0,
        "elinewidth": 1,
        "capsize": 2,
        "markersize": 2.5,
    }
    axes.errorbar(
        df[x],
        df[y],
        yerr=2 * df[std],
        color=color,
        label=label,
        **kwargs,
    )


def plot_mut_effs_posterior(axes, contrasts):
    peaks_contrasts = ["in" in x for x in contrasts.index]
    peaks_contrasts = contrasts.loc[peaks_contrasts, :].copy()
    items = [x.split("_") for x in peaks_contrasts.index.values]
    peaks_contrasts["mutation"] = [x[0] for x in items]
    peaks_contrasts["background"] = [x[-1] for x in items]

    bc1 = peaks_contrasts["background"] == "UUAAGGAGC"
    df = peaks_contrasts.loc[bc1, :].copy()
    df["x"] = np.arange(df.shape[0]) - 0.125
    plot_posterior(
        axes,
        df,
        color="grey",
        label=r"UUA$\bf{AGGAG}$C",
    )
    bc2 = peaks_contrasts["background"] == "UAAGGAGCA"
    df = peaks_contrasts.loc[bc2, :].copy()
    df["x"] = np.arange(df.shape[0]) + 0.125
    plot_posterior(
        axes,
        df,
        color="black",
        label=r"UA$\bf{AGGAG}$CA",
    )

    axes.set(
        xlabel="Mutation",
        ylabel="$\Delta$log(GFP)",
        xticks=df["x"],
        xticklabels=df["mutation"],
    )
    axes.axhline(0, linestyle="--", c="grey", lw=0.75)
    axes.legend(loc=(-0.02, 1.025), ncol=2)


def plot_path_posterior(axes, contrasts):
    seqs = ["AGGAGGNNN", "NGGAGGAGN", "NNNAGGNNN", "NNNAGGAGG"]
    labels = [
        r"$\bf{AGGAGG}$NNN",
        r"N$\bf{GGAGGAG}$N",
        r"NNN$\bf{AGG}$NNN",
        r"NNN$\bf{AGGAGG}$",
    ]
    path_contrasts = contrasts.loc[seqs, :].copy()
    path_contrasts["step"] = np.arange(1, path_contrasts.shape[0] + 1)
    plot_posterior(axes, path_contrasts, "step", "estimate", "std")
    axes.set(
        xlabel="Genetic background",
        ylabel="log(GFP)",
        xticks=path_contrasts["step"],
        xlim=(0.5, 4.5),
    )
    axes.set_xticklabels(labels, rotation=45, ha="right")

    if "NNNNNNNNN" in contrasts.index:
        axes.axhline(
            contrasts.loc["NNNNNNNNN", "estimate"],
            linestyle="--",
            c="grey",
            lw=0.75,
            label="Average",
        )
        axes.axhspan(
            ymin=contrasts.loc["NNNNNNNNN", "ci_95_lower"],
            ymax=contrasts.loc["NNNNNNNNN", "ci_95_upper"],
            color="grey",
            alpha=0.3,
            lw=0,
        )
    else:
        axes.axhline(
            0.0,
            linestyle="--",
            c="grey",
            lw=0.75,
            label="Average",
        )

    if "freqs" in contrasts.index:
        axes.axhline(
            contrasts.loc["freqs", "estimate"],
            linestyle="--",
            c="black",
            lw=0.75,
            label="AAGGAGGUG",
        )
        axes.axhspan(
            ymin=contrasts.loc["freqs", "ci_95_lower"],
            ymax=contrasts.loc["freqs", "ci_95_upper"],
            color="grey",
            alpha=0.2,
            lw=0,
        )
    elif "AAGGAGGUG" in contrasts.index:
        axes.axhline(
            contrasts.loc["AAGGAGGUG", "estimate"],
            linestyle="--",
            c="black",
            lw=0.75,
            label="AAGGAGGUG",
        )
        axes.axhspan(
            ymin=contrasts.loc["AAGGAGGUG", "ci_95_lower"],
            ymax=contrasts.loc["AAGGAGGUG", "ci_95_upper"],
            color="grey",
            alpha=0.2,
            lw=0,
        )
