import matplotlib
import gpmap.plot.mpl as mplot
import pandas as pd
import matplotlib.pyplot as plt
from gpmap.utils import read_edges
from code.plot_utils import (
    FIG_WIDTH,
    add_vcregression_labels,
    plot_path,
    annotate_seq,
    arrange_axis,
)


if __name__ == "__main__":
    matplotlib.use("Agg")

    # mfs = [0.6, 0.7, 0.8, 0.85]
    mfs = [-3, -2, -1, 0, 1]
    lims = (-5.5, 3.0)
    ticks = [-4, -3, -2, 1, 0, 1, 2, 3, 4]
    nplots = len(mfs)

    print("Load input data")
    edges_df = read_edges("results/edges.npz")
    fname = "results/intron.ler.map.mf_{}.nodes.pq"
    nodes_df = {mf: pd.read_parquet(fname.format(mf)) for mf in mfs}

    fig, subplots = plt.subplots(1, 4, figsize=(FIG_WIDTH, FIG_WIDTH / nplots))
    cbar_ax = subplots[1].inset_axes((-0, 0.7, 0.03, 0.3))
    vmin, vmax = 0, 1
    print("Plotting nodes")
    for mf, axes in zip(mfs, subplots):
        print("\tMean function at stationarity: {:.2f}".format(mf))
        df = nodes_df[mf]
        mplot.plot_edges(
            axes, df, edges_df=edges_df, alpha=0.02, rasterized=True
        )
        mplot.plot_nodes(
            axes,
            df,
            cbar_label="log(GFP)",
            cbar=True,
            cbar_axes=cbar_ax,
            cbar_orientation="vertical",
            sort_by="3",
            sort_ascending=True,
            size=1.5,
            vmin=vmin,
            vmax=vmax,
            rasterized=True,
        )
        arrange_axis(
            axes,
            x="1",
            y="2",
            ticks=ticks,
            lims=lims,
            fontsize=7,
            xpos=0.5,
            ypos=0.45,
            ms=3,
        )
        axes.set(aspect="equal", xlim=(-5.5, 3), ylim=(-4, 4.5))
        axes.set_title("Average log(GFP)={:.2f}".format(mf), fontsize=8)
    cbar_ax.set_ylabel("log(GFP)", fontsize=6)
    cbar_ax.set_yticklabels(cbar_ax.get_yticklabels(), fontsize=6)

    axes = subplots[-1]

    fig.tight_layout(w_pad=0.1)
    fig.savefig("figures/vcregression_visualization_mfs.png", dpi=300)
    fig.savefig("figures/vcregression_visualization_mfs.svg", dpi=300)
