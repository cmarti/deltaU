from code.plot_utils import (
    FIG_WIDTH,
    POSITION_LABELS,
    add_panel_labels,
    apply_plot_style,
    plot_pred_vs_obs_corr,
    plot_test_pred_comparison,
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from itertools import combinations, product
from gpmap.plot.mpl import (
    plot_correlation_U_sites,
    plot_interaction_matrix,
    plot_site_pairs_variance_components,
    plot_sites_variance_components,
)

if __name__ == "__main__":
    dataset_name = "intron.30C"
    position_labels = POSITION_LABELS[dataset_name]
    apply_plot_style()

    print(f"Plotting model fit for {dataset_name} dataset")

    print("Loading data for plotting")
    print("  Loading correlation data...")
    nodes_df = pd.read_csv(
        f"results/{dataset_name}.corrs.csv",
        dtype={"seq": str},
        index_col="seq",
    )

    print("  Loading a matrix...")
    a_matrix_inv = pd.read_csv(
        f"results/{dataset_name}.interaction_strength.csv", index_col=0
    )
    
    print("  Computing expected average squared epistatic coefficient under the prior...")
    a_matrix = 1. / a_matrix_inv
    sites = a_matrix.columns
    a_values = {(i, j): a_matrix.loc[int(i), j] for i, j in combinations(sites, 2)}
    Us = [tuple(sites[np.array(x)].values) for x in product([False, True], repeat=sites.shape[0])if np.sum(x) > 1]
    expected = {pair: 0 for pair in a_values.keys()}
    
    alpha = 4
    l = 8
    size_U = 2
    am1_over_alpha = (alpha - 1.) / alpha
    constant = 1. / ((alpha ** l) / 2 ** size_U * am1_over_alpha ** size_U)
    
    for U in Us:
        U_a = [a_values[tuple(pair)] for pair in combinations(U, 2)]
        lda_U = 1./np.sum(U_a)
        for pair in combinations(U, 2):
            expected[tuple(pair)] += constant * am1_over_alpha ** len(U) * lda_U
    
    m = []
    for (i, j), msec in expected.items():
        m.append({'i': i, 'j': j, 'v': np.sqrt(msec)})
    m = pd.DataFrame(m).pivot(index='i', columns='j', values='v')
    print(m)
    m = m.reindex(sites).T.reindex(sites).T.fillna(0)
    m = m + m.T
    
    print("Making figure...")
    fig, subplots = plt.subplots(1, 2, figsize=(0.5 * 1.12*FIG_WIDTH, 0.5 * FIG_WIDTH * 0.45))

    print("  Plotting a matrix...")
    axes = subplots[0]
    plot_interaction_matrix(
        a_matrix_inv,
        axes,
        vmax=None,
        position_labels=POSITION_LABELS[dataset_name],
        cbar_label='Interaction strength ($1/a_{ij}$)'
    )
    
    print("  Plotting root expected mean squared epistatic coefficients...")
    axes = subplots[1]
    plot_interaction_matrix(
        m,
        axes,
        vmax=None,
        position_labels=POSITION_LABELS[dataset_name],
        cbar_label=r'$\sqrt{\mathbb{E}\left[\,\overline{\epsilon_{ij}^2}\,\right]}$'
    )
    

    print("  Saving figure...")
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.8)
    fig.savefig("figures/intron.ler_prior.png", dpi=300)
    fig.savefig("figures/intron.ler_prior.svg", dpi=300)

    print("Done.")
