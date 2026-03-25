import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import gpmap.plot.mpl as mplot
from itertools import combinations
from code.plot_utils import FIG_WIDTH, apply_plot_style, plot_local_landscape




if __name__ == "__main__":
    dataset_name = "intron.30C"
    apply_plot_style()

    print(f"Calculating visualization for {dataset_name} dataset")

    print("  Loading inferred landscape...")
    data = pd.read_csv(
        f"results/{dataset_name}.ler.landscape.csv", index_col=0
    )
    data.index = [x.replace("T", "U") for x in data.index]
    data["19,20"] = [x[-3] + x[-2] for x in data.index]
    means = data.groupby(["19,20"])['f'].mean()

    fig, subplots = plt.subplots(
        2,
        2,
        figsize=(0.33 * FIG_WIDTH, 0.3 * FIG_WIDTH),
        sharex=True,
        sharey='row',
    )
    subplots = subplots.flatten()

    seqs_list = [
        ["GC", "GG", "CC", "CG"],
        ["AU", "AA", "UU", "UA"],
        ["UA", "UG", "UC", "GA", "GG", "GC"],
        ["AU", "GU", "CU", "AG", "GG", "CG"],
    ]
    
    for seqs, axes in zip(seqs_list, subplots):
        plot_local_landscape(means, seqs, axes, pos1="$_{19}$", pos2="$_{20}$")
        axes.set(
            xlim=(-0.5, 2.5), 
            # ylim=(-3 if len(xs) == 4 else -1.5, 1), 
            ylim=(-3, 1), 
            yticks=[-2, -1, 0],
            xlabel="", ylabel="", xticks=[]
        )
    
    print("  Saving figure...")
    fig.supxlabel("Genotype at positions 19 and 20", fontsize=7, x=0.6, y=0.08)
    fig.supylabel("Average fitness", fontsize=7, y=0.55, x=0.1)
    fig.tight_layout(w_pad=0.5, h_pad=0.5)
    fig.savefig(f"figures/{dataset_name}.local_landscapes.19-20.png", dpi=300)
    fig.savefig(f"figures/{dataset_name}.local_landscapes.19-20.svg", dpi=300)
    print("Done.")
