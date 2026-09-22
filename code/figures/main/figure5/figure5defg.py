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
        f"results/{dataset_name}.ssVC.map.mf_{mf}.nodes.pq"
    )
    nodes_df.index = [x.replace("T", "U") for x in nodes_df.index]
    edges_df = read_edges(f"results/{dataset_name}.edges.npz")

    print("  Calculating mutational effects")
    mutations = {
        "G2C": (0, 'G', 'C'),
        "C21G": (7, 'C', 'G'),
    }
    seqs = np.array([[c for c in x] for x in nodes_df.index])
    for mutation, (pos, a1, a2) in mutations.items():
        s1, s2 = seqs.copy(), seqs.copy()
        s1[:, pos] = a1
        s2[:, pos] = a2
        s1 = np.array(["".join(x) for x in s1])
        s2 = np.array(["".join(x) for x in s2])
        nodes_df[mutation] = nodes_df.loc[s2, 'function'].values - nodes_df.loc[s1, 'function'].values
    
    print("  Calculating epistatic coefficients")
    epistatic_coeffs = {
        "G3C_C20G": (1, 6, 'G', 'C', 'C', 'G'),
        "A3U_U20A": (1, 6, 'A', 'U', 'U', 'A'),
    }
    seqs = np.array([[c for c in x] for x in nodes_df.index])
    for epistatic_coeff, (i, j, a1_i, a2_i, a1_j, a2_j) in epistatic_coeffs.items():
        e = np.zeros(nodes_df.shape[0])
        signs = [1, -1, -1, 1]
        for sign, (a_i, a_j) in zip(signs, [(a1_i, a1_j), (a1_i, a2_j), (a2_i, a1_j), (a2_i, a2_j)]):
            s = seqs.copy()
            s[:, i] = a_i
            s[:, j] = a_j
            s = np.array(["".join(x) for x in s])
            e += sign * nodes_df.loc[s, 'function'].values
        nodes_df[epistatic_coeff] = e
    
    print("  Plotting visualization")
    print("    Plotting edges")
    dsg = dplot.plot_edges(
        nodes_df, edges_df=edges_df, resolution=800, x=x, y=y
    )
    fig = dplot.dsg_to_fig(dsg + dsg + dsg + dsg)
    # fig.set_size_inches((FIG_WIDTH, FIG_WIDTH * 0.3))

    cmap = "coolwarm"
    columns = ["G2C", "C21G", "G3C_C20G", "A3U_U20A"]
    for axes, label in zip(fig.axes, columns):
        if "_" in label:
            cbar_label = f"Epistatic coefficient\n{label.replace('_', '-')}"
        else:
            cbar_label = f"Mutational effect\n{label}"
            
        print(f"    Coloring by {label}")
        legendx, legendy = -0.05, 0.25
        nodes_hist_axes = axes.inset_axes((legendx, legendy - 0.125, 0.25, 0.1))
        nodes_cbar_axes = axes.inset_axes((legendx, legendy - 0.15, 0.25, 0.02))

        vmin, vmax = -6, 6
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
    fname = "figures/figure5defg"
    fig.savefig(f"{fname}.png", dpi=300)
    fig.savefig(f"{fname}.svg", dpi=300)
    print("Done.")
