import gpmap.plot.ds as dplot
import gpmap.plot.mpl as mplot
import gpmap.plot.ply as pplot
import pandas as pd
import matplotlib

from gpmap.utils import read_edges
from code.figures.plot_utils import (
    annotate_seq,
    plot_path,
    arrange_axis,
    plot_function_hist,
    FIG_WIDTH,
)


if __name__ == "__main__":
    matplotlib.use("Agg")

    print("Loading input data")
    nodes_df = pd.read_parquet("results/vcregression.map.mf_0.8.nodes.pq")
    edges_df = read_edges("results/edges.npz")

    wt = nodes_df.loc["GCCACC", "function"]
    print("Wild-type log(GFP) = {:.2f}".format(wt))

    print("Plotting VC regression visualization")
    dsg = dplot.plot_edges(nodes_df, edges_df=edges_df, resolution=800)
    fig = dplot.dsg_to_fig(dsg)
    fig.set_size_inches((FIG_WIDTH * 0.4, FIG_WIDTH * 0.4))
    axes = fig.axes[0]

    nodes_hist_axes = axes.inset_axes((0.8, 0.88, 0.3, 0.1))
    nodes_cbar_axes = axes.inset_axes((0.8, 0.85, 0.3, 0.02))

    vmin, vmax = 0, 1
    mplot.plot_nodes(
        axes,
        nodes_df,
        sort_by="3",
        sort_ascending=True,
        size=3,
        vmin=vmin,
        vmax=vmax,
        cmap="viridis",
        cbar_axes=nodes_cbar_axes,
        cbar_label="log(GFP)",
        cbar_orientation="horizontal",
        rasterized=True,
    )

    plot_function_hist(nodes_df, vmin, vmax, nodes_hist_axes, c="function")
    nodes_cbar_axes.set_xticklabels(
        nodes_cbar_axes.get_xticklabels(), fontsize=6
    )
    nodes_cbar_axes.set_xlabel("log(GFP)", fontsize=7)
    ticks = [-2.0, -1, 0, 1, 2, 3, 4]
    # lims = [-3-1, 3.]
    arrange_axis(axes, "1", "2", ticks, None, fontsize=8, xpos=0.48, ypos=0.36)
    axes.set(
        xticks=ticks,
        yticks=ticks,
        # ylim=(-3.0, 3.5),
        # xlim=(-2.25, 4.5),
        aspect="equal",
    )

    # fig.tight_layout()
    fig.savefig("figures/vcregression_visualization.png", dpi=300)
    fig.savefig("figures/vcregression_visualization.svg", dpi=600)

    pplot.plot_visualization(
        nodes_df,
        edges_df=edges_df,
        z="3",
        fpath="figures/vcregression_visualization",
    )
