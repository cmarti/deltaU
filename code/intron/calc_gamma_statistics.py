from code.plot_utils import POSITION_LABELS

import numpy as np
import pandas as pd
from gpmap.summary import GPmapSummarizer
from itertools import combinations

if __name__ == "__main__":
    dataset_name = "intron.30C"
    model_label = "ssVC"
    positions_labels = POSITION_LABELS[dataset_name]
    positions = np.arange(len(positions_labels))
    print("Calculating variance components for MAP estimate")

    print("  Loading MAP estimate..")
    data = pd.read_csv(
        f"results/{dataset_name}.{model_label}.landscape.csv", index_col=0
    )

    print("  Calculating summary statistics")
    s = GPmapSummarizer(n_alleles=4, seq_length=8, f=data["f"].values)
    
    print("    Calculating gamma_i_to_j statistics")
    gamma_i_to_j = s.calc_gamma_i_to_j()
    gamma_i_to_j["site_i"] = [positions_labels[i] for i in gamma_i_to_j["site_i"]]
    gamma_i_to_j["site_j"] = [positions_labels[i] for i in gamma_i_to_j["site_j"]]
    
    print("    Calculating gamma_i_to_jk statistics")
    S = list(range(8))
    records = []
    for i in S:
        D = [i]
        S_not_D = [i for i in S if i not in D]
        for j, k in combinations(S_not_D, 2):
            U = [j, k]
            gamma_UD = s.calc_gamma_U_D(U, D)
            cor_UD = s.calc_correlation_U_D(U, D)
            records.append(
                {
                    "i": positions_labels[i],
                    "j": positions_labels[j],
                    "k": positions_labels[k],
                    "D": ",".join([str(positions_labels[x]) for x in D]),
                    "U": ",".join([str(positions_labels[x]) for x in U]),
                    "gamma_UD": gamma_UD,
                    "cor_UD": cor_UD,
                }
            )
    gamma_i_to_jk = pd.DataFrame(records)
    
    print("    Calculating gamma_UD statistics for pairs of sites")
    S = list(range(8))
    records = []
    for U in combinations(S, 2):
        U_label = ','.join([str(positions_labels[i]) for i in U])
        S_not_U = [i for i in S if i not in U]
        for i, j in combinations(S_not_U, 2):
            D = [i, j]
            D_label = ','.join([str(positions_labels[i]) for i in D])
            gamma_UD = s.calc_gamma_U_D(U, D)
            cor_UD = s.calc_correlation_U_D(U, D)
            records.append(
                {
                    "U": U_label,
                    "D": D_label,
                    "gamma_UD": gamma_UD,
                    "cor_UD": cor_UD,
                }
            )
    gamma_UDs = pd.DataFrame(records)
    
    print("Saving gamma_i_to_j statistics")
    fpath = f"results/{dataset_name}.{model_label}.gamma_i_to_j.csv"
    gamma_i_to_j.to_csv(fpath)
    
    print("Saving gamma_i_to_jk statistics")
    fpath = f"results/{dataset_name}.{model_label}.gamma_i_to_jk.csv"
    gamma_i_to_jk.to_csv(fpath)
    
    print("Saving gamma_UD statistics for pairs of sites")
    fpath = f"results/{dataset_name}.{model_label}.gamma_UD_pairs.csv"
    gamma_UDs.to_csv(fpath)
    
    print("Done.")
