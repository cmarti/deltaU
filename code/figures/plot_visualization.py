import gpmap.plot.ds as dplot
import gpmap.plot.mpl as mplot
import gpmap.plot.ply as pplot
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from gpmap.utils import read_edges
from code.figures.plot_utils import (
    arrange_axis,
    plot_function_hist,
    FIG_WIDTH,
)


if __name__ == "__main__":
    method = 'ler'
    mf = -1.5
    matplotlib.use("Agg")
    x, y = 1,4

    print("Loading input data")
    nodes_df = pd.read_parquet(f"results/intron.{method}.map.mf_{mf}.nodes.pq")
    nodes_df['3'] = -nodes_df['3']
    print(nodes_df.max())
    print(nodes_df.min())
    edges_df = read_edges("results/edges.npz")

    print("Plotting visualization")
    # dsg = dplot.plot_edges(nodes_df, edges_df=edges_df, resolution=800)
    # fig = dplot.dsg_to_fig(dsg)
    # fig.set_size_inches((FIG_WIDTH * 0.4, FIG_WIDTH * 0.4))
    # axes = fig.axes[0]
    fig, axes = plt.subplots(1, 1, figsize=(FIG_WIDTH * 0.4, FIG_WIDTH * 0.4))

    nodes_hist_axes = axes.inset_axes((0.8, 0.88, 0.3, 0.1))
    nodes_cbar_axes = axes.inset_axes((0.8, 0.85, 0.3, 0.02))

    vmin, vmax = -7.5, 1
    mplot.plot_nodes(
        axes,
        nodes_df,
        x=str(x), 
        y=str(y),
        sort_by="7",
        sort_ascending=False,
        size=1,
        vmin=vmin,
        vmax=vmax,
        cmap="viridis",
        cbar_axes=nodes_cbar_axes,
        cbar_label="log(GFP)",
        cbar_orientation="horizontal",
        rasterized=True,
    )
    
    # boxes = [[(1, 2), (-2, -1)],
    #          [(1, 2), (-0.5, 0.5)],
    #         [(-2, -1), (-2, -1)],
    #         [(-2, -1), (-0.5, 0.5)],
    #         [(-2, 2), (1, 2)]]
    boxes = [[(1.5, 3), (-2, -0.5)],
            [(-2, 2), (1, 2)],
            [(-2, -0.5), (-2, -0.5)],
            ]
    
    for i, (xlims, ylims) in enumerate(boxes):
        mplot.plot_genotypes_box(axes, xlims, ylims, title='Region {}'.format(i+1),
                                title_pos='top' if i == 0 else 'right')

    plot_function_hist(nodes_df, vmin, vmax, nodes_hist_axes, c="function")
    nodes_cbar_axes.set_xticklabels(
        nodes_cbar_axes.get_xticklabels(), fontsize=6
    )
    nodes_cbar_axes.set_xlabel("log(GFP)", fontsize=7)
    ticks = [-2.0, -1, 0, 1, 2, 3, 4]
    # lims = [-3-1, 3.]
    arrange_axis(axes, str(x), str(y), ticks, None, fontsize=8, xpos=0.48, ypos=0.36)
    axes.set(
        # xticks=ticks,
        # yticks=ticks,
        ylim=(-3.0, 3),
        xlim=(-3.0, 3),
        aspect="equal",
    )

    # fig.tight_layout()
    fig.savefig(f"figures/intron.{method}.visualization.png", dpi=300)
    fig.savefig(f"figures/intron.{method}.visualization.svg", dpi=600)
    
    pplot.plot_visualization(
        nodes_df,
        # edges_df=edges_df,
        x='3',
        y='5',
        z="6",
        fpath=f"figures/intron.{method}.visualization",
    )
