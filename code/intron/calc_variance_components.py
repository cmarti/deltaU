from code.plot_utils import POSITION_LABELS

import numpy as np
import pandas as pd
from gpmap.summary import GPmapSummarizer

if __name__ == "__main__":
    dataset_name = "intron.30C"
    positions_labels = POSITION_LABELS[dataset_name]
    positions = np.arange(len(positions_labels))
    print("Calculating variance components for MAP estimate")

    print("  Loading MAP estimate..")
    data = pd.read_csv(
        f"results/{dataset_name}.ler.landscape.csv", index_col=0
    )

    print("  Calculating summary statistics")
    s = GPmapSummarizer(n_alleles=4, seq_length=8, f=data["f"].values)
    rmsec = s.calc_U_root_mean_squared_epistatic_coeffs()
    v_k_vcs = s.calc_V_k_variance_components()
    v_U_vcs = s.calc_V_U_variance_components()
    sites = s.calc_sites_variance_perc(v_U_vcs)
    sites.columns = positions_labels
    pairs_var = s.calc_site_pairs_variance_perc(v_U_vcs)

    pairs_pw = s.calc_site_pairs_variance_perc(
        v_U_vcs.loc[v_U_vcs["k"] == 2, :]
    )
    pairs_ho = s.calc_site_pairs_variance_perc(v_U_vcs, min_k=3)
    pairs = s.calc_site_pairs_variance_perc(v_U_vcs)
    m_bottom = pairs_pw.pivot(
        index="site1", columns="site2", values="variance_perc"
    )
    m_bottom = m_bottom.reindex(positions).T.reindex(positions).fillna(0)
    m_top = pairs_ho.pivot(
        index="site1", columns="site2", values="variance_perc"
    )
    m_top = m_top.reindex(positions).T.reindex(positions).fillna(0).T
    m = m_top + m_bottom
    m.index = positions_labels
    m.columns = positions_labels
    
    print("Saving MAP variance components")
    print("  Variance explained by interactions of order k")
    fpath = f"results/{dataset_name}.ler.variance_k.csv"
    v_k_vcs.to_csv(fpath)
    
    print("  Variance explained by interactions of order k for site i")
    fpath = f"results/{dataset_name}.ler.sites_variance_k.csv"
    sites.iloc[::-1, :].to_csv(fpath)

    print(
        "  Variance explained by interactions of order k=2 and k>2 for pairs of sites"
    )
    fpath = f"results/{dataset_name}.ler.sites_pairs_variance.csv"
    m.to_csv(fpath)
    
    print("Saving RMS epistatic coefficients")
    m = rmsec.pivot(
        index="site1", columns="site2", values="rmsec"
    )
    m = m.reindex(positions).T.reindex(positions).fillna(0)
    m = m + m.T
    m.index = positions_labels
    m.columns = positions_labels
    fpath = f"results/{dataset_name}.ler.rmsec.csv"
    m.to_csv(fpath)
    print("Done.")
