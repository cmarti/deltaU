import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from gpmap.summary import GPmapSummarizer
from code.plot_utils import apply_plot_style, FIG_WIDTH

if __name__ == "__main__":
    dataset_name = 'intron.30C'
    apply_plot_style()
    print("Calculating variance components for MAP estimate")
    
    print("Loading data for plotting")
    print("  Loading MAP estimate..")
    data = pd.read_csv(f"results/{dataset_name}.ler.landscape.csv", index_col=0)
    
    
    
    

    print("  Calculating variance components")
    s = GPmapSummarizer(n_alleles=4, seq_length=8, f=data["f"].values)
    v_U_vcs = s.calc_V_U_variance_components()
    sites = s.calc_sites_variance_perc(v_U_vcs)
    pairs_pw = s.calc_site_pairs_variance_perc(v_U_vcs.loc[v_U_vcs['k'] == 2, :])
    pairs_ho = s.calc_site_pairs_variance_perc(v_U_vcs, min_k=3)
    pairs = s.calc_site_pairs_variance_perc(v_U_vcs)

    print("  Creating figure")
    orders = np.arange(1, 9)
    positions = [2, 3, 4, 5, 18, 19, 20, 21]
    ticks = np.arange(8)

    fig, subplots = plt.subplots(1, 2, figsize=(0.6*FIG_WIDTH, 0.25*FIG_WIDTH))
    
    axes = subplots[0]
    print('    Plotting site variance components...')
    im = axes.imshow(
        sites.iloc[::-1, :],
        cmap="Greys",
        vmin=0,
        vmax=15,
    )
    axes.set(
        xticks=ticks,
        xticklabels=positions,
        xlabel="Site",
        yticks=ticks,
        yticklabels=orders[::-1],
        ylabel="Interaction order $k$",
        aspect="equal",
    )
    plt.colorbar(im, shrink=0.8, label="% variance explained")
    
    print('    Plotting pair variance components...')
    m_bottom = pairs_pw.pivot(index="site1", columns="site2", values="variance_perc")
    m_bottom = m_bottom.reindex(ticks).T.reindex(ticks).fillna(0)
    m_top = pairs_ho.pivot(index='site1', columns='site2', values='variance_perc')
    m_top = m_top.reindex(ticks).T.reindex(ticks).fillna(0).T
    m = m_top + m_bottom
    
    # m = pairs.pivot(index='site1', columns='site2', values='variance_perc')
    # m = m.reindex(ticks).T.reindex(ticks).fillna(0)
    # m = m + m.T
    
    axes = subplots[1]
    im = axes.imshow(
        m,
        cmap="Greys",
        vmin=0,
        vmax=60,
    )
    axes.set(
        xticks=ticks,
        xlabel="Site 1",
        yticks=ticks,
        xticklabels=positions,
        yticklabels=positions,
        ylabel="Site 2",
        aspect="equal",
    )
    plt.colorbar(im, shrink=0.8, label="% pairwise and higher-order\nvariance explained")
    
    print("  Saving figure...")
    fig.tight_layout()
    fig.savefig(f"figures/{dataset_name}.map_variance_components.png", dpi=300)
    
    print("Done.")
