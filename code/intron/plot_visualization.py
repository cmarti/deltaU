import numpy as np
import pandas as pd
import logomaker as lm

import gpmap.plot.ds as dplot
import gpmap.plot.mpl as mplot

from gpmap.genotypes import get_genotypes_from_region
from gpmap.utils import read_edges
from code.plot_utils import (
    FIG_WIDTH,
    apply_plot_style,
    arrange_axis,
    POSITION_LABELS,
)


def plot_function_hist(ndf, vmin, vmax, nodes_hist_axes, c):
    bins = np.linspace(vmin, vmax, 30)
    mplot.plot_color_hist(nodes_hist_axes, ndf[c], cmap="viridis", bins=bins)
    nodes_hist_axes.set_ylabel("Frequency", fontsize=7)


if __name__ == "__main__":
    wt = ["AGGUACAU"]
    apply_plot_style()
    dataset_name = "intron.30C"
    position_labels = POSITION_LABELS[dataset_name]
    mf = 1.6
    x, y, z = "1", "2", "3"
    print(f"Plotting visualization for {dataset_name} dataset")

    print("  Loading input data")
    nodes_df = pd.read_parquet(
        f"results/{dataset_name}.ler.map.mf_{mf}.nodes.pq"
    )
    nodes_df.index = [x.replace("T", "U") for x in nodes_df.index]
    edges_df = read_edges(f"results/{dataset_name}.edges.npz")

    print("  Plotting visualization")
    dsg = dplot.plot_edges(
        nodes_df, edges_df=edges_df, resolution=800, x=x, y=y
    )
    fig = dplot.dsg_to_fig(dsg)
    # fig.set_size_inches((FIG_WIDTH * 0.66, FIG_WIDTH * 0.66))
    axes = fig.axes[0]
    # fig, axes = plt.subplots(1, 1, figsize=(FIG_WIDTH * 0.6, FIG_WIDTH * 0.6))

    legendx, legendy = -0.05, 0.25
    nodes_hist_axes = axes.inset_axes((legendx, legendy - 0.125, 0.25, 0.1))
    nodes_cbar_axes = axes.inset_axes((legendx, legendy - 0.15, 0.25, 0.02))

    vmin, vmax = -5, 4
    mplot.plot_nodes(
        axes,
        nodes_df,
        x=str(x),
        y=str(y),
        sort_by=str(z),
        sort_ascending=False,
        size=1.5,
        vmin=vmin,
        vmax=vmax,
        cmap="viridis",
        cbar_axes=nodes_cbar_axes,
        cbar_orientation="horizontal",
        cbar_label="Fitness",
        rasterized=True,
    )
    mplot.plot_nodes(
        axes,
        nodes_df.loc[wt, :],
        x=str(x),
        y=str(y),
        size=6,
        lw=0.75,
        vmin=vmin,
        vmax=vmax,
        cbar=False,
        cmap="viridis",
        rasterized=True,
    )

    plot_function_hist(nodes_df, vmin, vmax, nodes_hist_axes, c="function")
    nodes_hist_axes.set_facecolor("none")
    nodes_cbar_axes.set(xticks=[-4, -2, 0, 2, 4])
    nodes_cbar_axes.set_xticklabels([-4, -2, 0, 2, 4], fontsize=6)
    nodes_cbar_axes.set_xlabel("Fitness", fontsize=7)

    arrange_axis(
        axes,
        x=x,
        y=y,
        ticks=np.arange(-2, 4),
        lims=(-2, 4),
        fontsize=7,
        xpos=0.43,
        ypos=0.41,
    )

    boxes = [
        [(1.5, 3), (0.3, 2.25)],
        [(-2, -1), (1, 2.25)],
        [(-0.65, 0.75), (-1.9, -1)],
    ]

    xpos = [0.67, 0.0, 0.64]
    ypos = [0.92, 0.92, 0.1]

    for i, (xlims, ylims) in enumerate(boxes):
        mplot.plot_genotypes_box(
            axes,
            xlims,
            ylims,
            # title="Region {}".format(i + 1),
            # title_pos="right" if i == 1 else "top",
            lw=0.5,
            fontsize=6,
            c="grey",
        )
        seqs = get_genotypes_from_region(
            nodes_df,
            min_values={x: xlims[0], y: ylims[0], "function": -1},
            max_values={x: xlims[1], y: ylims[1]},
        )
        m = lm.alignment_to_matrix(
            seqs.values, to_type="probability", pseudocount=0
        )
        m.index = np.arange(m.shape[0])

        logo_axes = axes.inset_axes((xpos[i], ypos[i], 0.275, 0.1))
        logo = lm.Logo(m, ax=logo_axes, vpad=0.05)
        logo_axes.set(
            ylabel="Probability",
            xlabel="Position",
            xticks=np.arange(m.shape[0]),
            xticklabels=position_labels,
            title="Region {}".format(i + 1),
        )

    axes.set(
        xlim=(-2.25, 3.5),
        ylim=(-2.25, 3.25),
        aspect="equal",
    )
    axes.margins(0.1)

    print("  Saving figure...")
    fig.tight_layout()
    fig.savefig(f"figures/{dataset_name}.ler.visualization.png", dpi=300)
    fig.savefig(f"figures/{dataset_name}.ler.visualization.svg", dpi=300)
    print("Done.")
