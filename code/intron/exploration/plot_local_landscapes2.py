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
    data["2,21"] = [x[0] + x[-1] for x in data.index]
    data["3,20"] = [x[1] + x[-2] for x in data.index]

    fig, subplots = plt.subplots(
        4,
        2,
        figsize=(0.33 * FIG_WIDTH, 0.3 * FIG_WIDTH * 1.8),
        sharex=True,
        sharey='row',
    )
    subplots = subplots.flatten()

    backgrounds = ['GC',  'CG', 'AU','UA', 'GU', 'GA', 'AA', 'GG']
    # backgrounds = ['GC', 'CG', 'CC', 'GG']
    # backgrounds = ['AU', 'AA', 'UU', 'UA']
    seqs = ['GC', 'GG', 'CC', 'CG']
    # seqs = ['AC', 'CC', 'AG', 'CG']
    # seqs = ['AU', 'AA', 'UU', 'UA']
    
    for background, axes in zip(backgrounds, subplots):
        label = background[0] + "$_{2}$" + background[1] + "$_{21}$ background"
        background_data = data.loc[data['2,21'] == background, :]
        print(background_data)
        means = background_data.groupby(["3,20"])['f'].mean()
        plot_local_landscape(means, seqs, axes, pos1="$_{3}$", pos2="$_{20}$")
        axes.text(0.05, 0.95, label, transform=axes.transAxes, ha='left', va='top', fontsize=5)
        axes.set(
            xlim=(-0.5, 2.5), 
            # ylim=(-3 if len(xs) == 4 else -1.5, 1), 
            ylim=(-3, 3.5), 
            # yticks=[-2, -1, 0],
            xlabel="", ylabel="", xticks=[]
        )
    
    print("  Saving figure...")
    fig.supxlabel("Genotype at positions 3 and 20", fontsize=7, x=0.6, y=0.08)
    fig.supylabel("Average fitness", fontsize=7, y=0.55, x=0.1)
    fig.tight_layout(w_pad=0.5, h_pad=0.5)
    fig.savefig(f"figures/{dataset_name}.local_landscapes.2-3-20-21.png", dpi=300)
    fig.savefig(f"figures/{dataset_name}.local_landscapes.2-3-20-21.svg", dpi=300)
    print("Done.")
