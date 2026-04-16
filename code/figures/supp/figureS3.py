from code.plot_utils import (
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
    x, y, z = "1", "2", "3"
    mean_functions = [0, 0.4, 0.8, 1.2, 1.6, 1.8]
    print(f"Plotting visualization for {dataset_name} dataset")

    print("  Loading edges")
    edges_df = read_edges(f"results/{dataset_name}.edges.npz")
    
    print("  Loading visualization coordinates under different mean functions")
    nodes_dfs = {}
    for mf in mean_functions:
        nodes_df = pd.read_parquet(
            f"results/{dataset_name}.ler.map.mf_{mf}.nodes.pq"
        )
        nodes_dfs[mf] = nodes_df
    nodes_dfs[0]['2'] = -nodes_dfs[0]['2']
    
    print("  Plotting edges")
    dsg = None
    for mf, nodes_df in nodes_dfs.items():
        print(f"    For visualization under a mean function of {mf}")
        if dsg is None:
            dsg = dplot.plot_edges(
                nodes_df, edges_df=edges_df, resolution=800, x=x, y=y
            )
        else:
            dsg = dsg + dplot.plot_edges(
                nodes_df, edges_df=edges_df, resolution=800, x=x, y=y
            )
    print('  Rendering edges')
    fig = dplot.dsg_to_fig(dsg.cols(3))
    # fig.set_size_inches((FIG_WIDTH, FIG_WIDTH * 0.66))
    
    print("  Plotting nodes")
    for axes, (mf, nodes_df) in zip(fig.axes, nodes_dfs.items()):
        print(f"    For visualization under a mean function of {mf}")
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

        axes.set(
            xlim=(-2.25, 3.5),
            ylim=(-2.25, 3.25),
            aspect="equal",
            title=f'Mean fitness = {mf:.1f}'
        )
        axes.margins(0.1)

    print("  Saving figure...")
    fig.tight_layout()
    fname = "figures/figureS3"
    fig.savefig(f"{fname}.png", dpi=300)
    fig.savefig(f"{fname}.svg", dpi=300)
    print("Done.")
