from code.plot_utils import (
    POSITION_LABELS,
    apply_plot_style,
    arrange_axis,
)

import gpmap.plot.mpl as mplot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gpmap.utils import read_edges
from matplotlib.patches import Patch


def plot_function_hist(ndf, vmin, vmax, nodes_hist_axes, c, cmap="viridis"):
    bins = np.linspace(vmin, vmax, 30)
    mplot.plot_color_hist(nodes_hist_axes, ndf[c], cmap=cmap, bins=bins)
    nodes_hist_axes.set_ylabel("Frequency", fontsize=7)

def style_visualization(axes):
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
    mapping = {"A": 0.2, "C": 0.6, "G": 0.4, "U": 0.8}
    for pos in range(8):
        nodes_df[position_labels[pos]] = [mapping[x[pos]] for x in nodes_df.index]

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

    fig, subplots = plt.subplots(8, 5, figsize=(16, 20))

    print("    Plotting nodes")
    cmap = "coolwarm"
    for p, ax_col in enumerate(subplots):
        
        print(f"    Coloring by alleles at position {position_labels[p]}")
        axes = ax_col[0]
        mplot.plot_nodes(
            axes,
            nodes_df,
            x=str(x),
            y=str(y),
            sort_by=str(z),
            sort_ascending=False,
            # sort_by="function",
            # sort_ascending=True,
            color=position_labels[p],
            cmap='magma',
            size=1.5,
            cbar=False,
            rasterized=True,
        )
        # Add categorical legend for magma colormap
        legend_elements = [Patch(facecolor=plt.cm.magma(mapping[allele]), label=allele) 
                  for allele in "ACGU"]
        axes.legend(handles=legend_elements, loc=4)
        style_visualization(axes)
        
        for allele, axes in zip("ACGU", ax_col[1:]):
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
            style_visualization(axes)

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
