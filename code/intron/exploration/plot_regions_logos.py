import gpmap.plot.ds as dplot
import gpmap.plot.mpl as mplot
import gpmap.plot.ply as pplot
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import logomaker as lm

from gpmap.genotypes import get_genotypes_from_region
from code.plot_utils import FIG_WIDTH, apply_plot_style


if __name__ == "__main__":
    apply_plot_style()
    dataset_label = "intron"
    print("Plotting logos representing different regions in the visualization")
    print(f"  Dataset: {dataset_label}")
    
    mf = 1
    x, y = 1, 2
    x, y = str(x), str(y)
    positions_labels = [2, 3, 4, 5, 18, 19, 20, 21]
    print(f"  Mean function at stationarity: {mf}")
    print(f"  Diffusion axes {x} and {y}")

    print("Loading visualization data...")
    nodes_df = pd.read_parquet(
        f"results/{dataset_label}.ler.map.mf_{mf}.nodes.pq"
    )

    print("  Defining visualization regions")
    boxes = [
        [(1.5, 3.25), (1, 2.5)],
        [(-1, 1), (-2, -1)],
        [(-2, -1), (1, 2.5)],
    ]
    peak_seqs = [
        get_genotypes_from_region(
            nodes_df,
            min_values={x: xs[0], y: ys[0], "function": -1},
            max_values={x: xs[1], y: ys[1]},
        )
        for xs, ys in boxes
    ]

    n_regions = len(boxes)
    print(f'Making figure for {n_regions} regions...')
    fig, subplots = plt.subplots(n_regions, 1, figsize=(2.5, 1.15 * n_regions))
    
    for i, (seqs, axes) in enumerate(zip(peak_seqs, subplots)):
        print(f"  Region {i}")
        m = lm.alignment_to_matrix(
            seqs.values, to_type="probability", pseudocount=0
        )
        m.index = np.arange(m.shape[0])
        logo = lm.Logo(m, ax=axes, vpad=0.05)
        axes.set(
            ylabel="Probability",
            xlabel="Position",
            xticks=np.arange(m.shape[0]),
            xticklabels=positions_labels,
            title="Region {}".format(i + 1),
        )
        
    print('Saving figure...')
    fig.tight_layout()
    fig.savefig(f"figures/{dataset_label}.logos.png", dpi=300)
    print('Done.')
