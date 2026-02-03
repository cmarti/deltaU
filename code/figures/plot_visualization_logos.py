import gpmap.plot.ds as dplot
import gpmap.plot.mpl as mplot
import gpmap.plot.ply as pplot
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import logomaker as lm

from gpmap.genotypes import get_genotypes_from_region
from code.figures.plot_utils import (
    arrange_axis,
    plot_function_hist,
    FIG_WIDTH,
)


if __name__ == "__main__":
    method = 'ler'
    mf = -1
    matplotlib.use("Agg")
    x, y = 1, 2
    positions_labels = [2, 3, 4, 5, 18, 19, 20, 21]
    seqs = ['AGGTACAT',
            'GCCCACGT',
            'GTTTACGT',
            'CAAATGCG',
            'GTTTACGC',
            'GTTTACGT',
            'ATTTACGT',
            'GTTTATGC',
            'GTTTAACG',
            
            # 'CTAAACAG',
            # 'GCGTACGT',
            # 'ACGTACGT',
            # 'ACGTACGT',
            ]

    print("Loading input data")
    nodes_df = pd.read_parquet(f"results/intron.{method}.map.mf_{mf}.nodes.pq")
    nodes_df['3'] = -nodes_df['3']
    print(nodes_df.loc[seqs, :])
    
    # boxes = [[(1, 2), (-2, -1)],
    #          [(1, 2), (-0.5, 0.5)],
    #         [(-2, 2), (1, 2)],
    #         [(-2, -1), (-0.5, 0.5)],
    #         [(-2, -1), (-2, -1)],
    #         ]
    
    boxes = [[(1.5, 3), (-2, -0.5)],
            [(-2, 2), (1, 2)],
            [(-2, -0.5), (-2, -0.5)],
            ]
    
    peak_seqs = [get_genotypes_from_region(nodes_df, min_values={'1': xs[0], '5': ys[0], 'function': -3},
                                        max_values={'1': xs[1], '5': ys[1]})
                for xs, ys in boxes]

    fig, subplots = plt.subplots(5, 1, figsize=(4, 7.5))
    for i, (seqs, axes) in enumerate(zip(peak_seqs, subplots)):
        m = lm.alignment_to_matrix(seqs.values, to_type='probability', pseudocount=0)
        m.index = np.arange(m.shape[0])
        logo = lm.Logo(m, ax=axes, vpad=0.05)
        axes.set(ylabel='Probability', xlabel='Position',
                            xticks=np.arange(m.shape[0]),
                            xticklabels=positions_labels, title='Region {}'.format(i+1))
    fig.tight_layout()
    fig.savefig('figures/intron.logos.png', dpi=300)