from code.plot_utils import (
    POSITION_LABELS,
    apply_plot_style,
    arrange_axis,
)

import gpmap.plot.ds as dplot
import gpmap.plot.mpl as mplot
import numpy as np
import pandas as pd
from gpmap.utils import read_edges


def plot_function_hist(ndf, vmin, vmax, nodes_hist_axes, c):
    bins = np.linspace(vmin, vmax, 30)
    mplot.plot_color_hist(nodes_hist_axes, ndf[c], cmap="viridis", bins=bins)
    nodes_hist_axes.set_ylabel("Frequency", fontsize=7)


if __name__ == "__main__":
    apply_plot_style()
    dataset_name = "intron.30C"
    position_labels = POSITION_LABELS[dataset_name]
    mf = 1.8
    x, y, z = "1", "2", "3"
    print(f"Plotting visualization for {dataset_name} dataset")

    print("  Loading input data")
    nodes_df = pd.read_parquet(
        f"results/{dataset_name}.ssVC.map.mf_{mf}.nodes.pq"
    )
    nodes_df.index = [x.replace("T", "U") for x in nodes_df.index]
    edges_df = read_edges(f"results/{dataset_name}.edges.npz")

    print("  Plotting visualization")
    dsg = dplot.plot_edges(
        nodes_df, edges_df=edges_df, resolution=800, x=x, y=y
    )
    fig = dplot.dsg_to_fig(dsg)
    axes = fig.axes[0]

    legendx, legendy = 0.0, 0.25
    nodes_hist_axes = axes.inset_axes((legendx, legendy - 0.125, 0.2, 0.1))
    nodes_cbar_axes = axes.inset_axes((legendx, legendy - 0.15, 0.2, 0.02))

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

    plot_function_hist(nodes_df, vmin, vmax, nodes_hist_axes, c="function")
    nodes_hist_axes.set_facecolor("none")
    nodes_cbar_axes.set(xticks=[-4, -2, 0, 2, 4])
    nodes_cbar_axes.set_xticklabels([-4, -2, 0, 2, 4], fontsize=6)
    nodes_cbar_axes.set_xlabel("Fitness", fontsize=7)

    arrange_axis(
        axes,
        x=x,
        y=y,
        ticks=np.arange(-2, 3),
        lims=(-2, 4),
        fontsize=7,
        xpos=0.465,
        ypos=0.415,
        ms=2,
    )

    axes.set(
        xticks=np.arange(-2, 4),
        yticks=np.arange(-1, 4),
        xlim=(-2.1, 3.25),
        ylim=(-1.9, 2.35),
        aspect="equal",
    )
    axes.margins(0.1)

    print("  Saving figure...")
    fig.tight_layout()
    fname = "figures/figure5a"
    fig.savefig(f"{fname}.png", dpi=300)
    fig.savefig(f"{fname}.svg", dpi=300)
    print("Done.")
