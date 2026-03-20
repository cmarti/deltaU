import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import gpmap.plot.ds as dplot
import gpmap.plot.mpl as mplot

from gpmap.utils import read_edges
from code.plot_utils import (
    FIG_WIDTH,
    apply_plot_style,
    arrange_axis,
    POSITION_LABELS,
)


def plot_function_hist(ndf, vmin, vmax, nodes_hist_axes, c, cmap="viridis"):
    bins = np.linspace(vmin, vmax, 30)
    mplot.plot_color_hist(nodes_hist_axes, ndf[c], cmap=cmap, bins=bins)
    nodes_hist_axes.set_ylabel("Frequency", fontsize=7)


if __name__ == "__main__":
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

    print("  Computing background-specific allelic effects")
    seqs_array = np.array([[c for c in x] for x in nodes_df.index])
    for position in range(8):
        alleles = seqs_array.copy()
        cols = []
        for allele in "ACGU":
            alleles[:, position] = allele
            seqs = np.array(["".join(x) for x in alleles])
            label = f"{position_labels[position]}{allele}"
            cols.append(label)
            nodes_df[label] = nodes_df["function"].reindex(seqs).values
        for col in cols:
            nodes_df[col] -= nodes_df[cols].mean(1).values

    print("  Plotting visualization")
    print("    Plotting edges")
    # dsg = dplot.plot_edges(
    #     nodes_df, edges_df=edges_df, resolution=800, x=x, y=y
    # )
    # for i in range(32):
    #     dsg += dsg
    # fig = dplot.dsg_to_fig(dsg.cols(8))

    fig, subplots = plt.subplots(8, 4, figsize=(12, 20))

    print("    Plotting nodes")
    cmap = "coolwarm"
    for p, ax_col in enumerate(subplots):
        for allele, axes in zip("ACGU", ax_col):
            label = f"{position_labels[p]}{allele}"
            print(f"      Coloring by {label}")
            legendx, legendy = -0.05, 0.25
            nodes_hist_axes = axes.inset_axes(
                (legendx, legendy - 0.125, 0.25, 0.1)
            )
            nodes_cbar_axes = axes.inset_axes(
                (legendx, legendy - 0.15, 0.25, 0.02)
            )

            vmin, vmax = -4, 4
            mplot.plot_nodes(
                axes,
                nodes_df,
                x=str(x),
                y=str(y),
                sort_by=str(z),
                sort_ascending=False,
                # sort_by="function",
                # sort_ascending=True,
                color=label,
                size=1.5,
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
                cbar_axes=nodes_cbar_axes,
                cbar_orientation="horizontal",
                rasterized=True,
            )

            plot_function_hist(
                nodes_df, vmin, vmax, nodes_hist_axes, c=label, cmap=cmap
            )
            nodes_hist_axes.set_facecolor("none")
            nodes_cbar_axes.set(xticks=[-4, -2, 0, 2, 4])
            nodes_cbar_axes.set_xticklabels([-4, -2, 0, 2, 4], fontsize=6)
            cbar_label = f"Allelic effect\n{label}"
            nodes_cbar_axes.set_xlabel(cbar_label, fontsize=7)
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
            )
            axes.margins(0.1)

    print("  Saving figure...")
    fig.tight_layout()
    fig.savefig(
        f"figures/{dataset_name}.ler.visualization_allelic_effects.png",
        dpi=300,
    )
    fig.savefig(
        f"figures/{dataset_name}.ler.visualization_allelic_effects.svg",
        dpi=300,
    )
    print("Done.")
