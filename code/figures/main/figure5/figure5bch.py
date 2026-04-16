from code.plot_utils import FIG_WIDTH, apply_plot_style, plot_local_landscape

import matplotlib.pyplot as plt
import pandas as pd

if __name__ == "__main__":
    dataset_name = "intron.30C"
    apply_plot_style()

    print(f"Plotting local landscapes for {dataset_name} dataset")

    print("  Loading contrasts results...")
    contrasts = pd.read_csv(
        f"results/{dataset_name}.ler.contrasts.csv", index_col=0
    )

    print("  Plotting interactions between positions 2 and 21")
    fig, subplots = plt.subplots(
        2,
        2,
        figsize=(0.33 * FIG_WIDTH, 0.3 * FIG_WIDTH),
        sharex=True,
        sharey="row",
    )
    subplots = subplots.flatten()

    seqs_list = [
        ["GC", "GG", "CC", "CG"],
        ["AU", "AA", "UU", "UA"],
        ["AC", "AG", "UC", "UG"],
        ["GU", "GA", "CU", "CA"],
    ]

    for seqs, axes in zip(seqs_list, subplots):
        rownames = [f"2{s[0]}_21{s[1]}" for s in seqs]
        c = contrasts.loc[rownames, :]
        plot_local_landscape(c, seqs, axes)
        axes.set(
            xlim=(-0.5, 2.5),
            ylim=(-3, 1),
            yticks=[-2, -1, 0],
            xlabel="",
            ylabel="",
            xticks=[],
        )

    print("    Saving figure...")
    fig.supxlabel("Genotype at positions 2 and 21", fontsize=7, x=0.6, y=0.08)
    fig.supylabel("Average fitness", fontsize=7, y=0.55, x=0.1)
    fig.tight_layout(w_pad=0.5, h_pad=0.5)
    fname = "figures/figure5b"
    fig.savefig(f"{fname}.png", dpi=300)
    fig.savefig(f"{fname}.svg", dpi=300)

    print(
        "  Plotting context-dependent interactions between positions 3 and 20"
    )
    fig, subplots = plt.subplots(
        2,
        2,
        figsize=(0.33 * FIG_WIDTH, 0.3 * FIG_WIDTH),
        sharex=True,
        sharey="row",
    )
    subplots = subplots.flatten()

    backgrounds = ["GC", "CG", "AU", "UA", "GU", "GA", "AA", "GG"]
    seqs = ["GC", "GG", "CC", "CG"]

    for background, axes in zip(backgrounds, subplots):
        label = background[0] + "$_{2}$" + background[1] + "$_{21}$ background"
        rownames = [
            f"2{background[0]}_3{s[0]}_20{s[1]}_21{background[1]}"
            for s in seqs
        ]
        c = contrasts.loc[rownames, :]
        plot_local_landscape(c, seqs, axes, pos1="$_{3}$", pos2="$_{20}$")
        axes.set(
            xlim=(-0.5, 2.5), ylim=(-3, 3.5), xlabel="", ylabel="", xticks=[]
        )
        axes.text(0.05, 0.95, label, transform=axes.transAxes, ha='left', va='top', fontsize=5)

    print("    Saving figure...")
    fig.supxlabel("Genotype at positions 3 and 20", fontsize=7, x=0.6, y=0.08)
    fig.supylabel("Average fitness", fontsize=7, y=0.55, x=0.1)
    fig.tight_layout(w_pad=0.5, h_pad=0.5)
    fname = "figures/figure5c"
    fig.savefig(f"{fname}.png", dpi=300)
    fig.savefig(f"{fname}.svg", dpi=300)
    
    print(
        "  Plotting context-dependent interactions between positions 19 and 20"
    )
    fig, subplots = plt.subplots(
        3,
        2,
        figsize=(0.33 * FIG_WIDTH, 0.42 * FIG_WIDTH),
        sharex=True,
        sharey='row',
    )
    subplots = subplots.flatten()

    backgrounds = ['GC',  'CG','AU', 'UA', 'AC', 'AG']
    seqs = ['GC', 'GG', 'CC', 'CG']

    for background, axes in zip(backgrounds, subplots):
        label = background[0] + "$_{2}$" + background[1] + "$_{21}$ background"
        rownames = [
            f"2{background[0]}_19{s[0]}_20{s[1]}_21{background[1]}"
            for s in seqs
        ]
        c = contrasts.loc[rownames, :]
        plot_local_landscape(c, seqs, axes, pos1="$_{19}$", pos2="$_{20}$")
        axes.set(
            xlim=(-0.5, 2.5), ylim=(-3, 3.5), xlabel="", ylabel="", xticks=[]
        )
        axes.text(0.05, 0.95, label, transform=axes.transAxes, ha='left', va='top', fontsize=5)

    print("    Saving figure...")
    fig.supxlabel("Genotype at positions 19 and 20", fontsize=7, x=0.6, y=0.08)
    fig.supylabel("Average fitness", fontsize=7, y=0.55, x=0.1)
    fig.tight_layout(w_pad=0.5, h_pad=0.5)
    
    fname = "figures/figure5h"
    fig.savefig(f"{fname}.png", dpi=300)
    fig.savefig(f"{fname}.svg", dpi=300)
    
    print("Done.")
