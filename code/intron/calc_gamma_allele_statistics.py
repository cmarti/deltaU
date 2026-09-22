from code.plot_utils import POSITION_LABELS

import numpy as np
import pandas as pd
from gpmap.summary import GPmapSummarizer


def add_labels(gamma, positions_labels, alphabet):
    gamma["A_i"] = [alphabet[i] for i in gamma["A_i"]]
    gamma["A_j"] = [alphabet[i] for i in gamma["A_j"]]
    gamma["B_i"] = [alphabet[i] for i in gamma["B_i"]]
    gamma["B_j"] = [alphabet[i] for i in gamma["B_j"]]
    gamma["site_i"] = [positions_labels[i] for i in gamma["site_i"]]
    gamma["site_j"] = [positions_labels[i] for i in gamma["site_j"]]
    gamma["mut_i"] = [
        f"{a1}{p}{a2}"
        for a1, p, a2 in zip(gamma["A_i"], gamma["site_i"], gamma["B_i"])
    ]
    gamma["mut_j"] = [
        f"{a1}{p}{a2}"
        for a1, p, a2 in zip(gamma["A_j"], gamma["site_j"], gamma["B_j"])
    ]


if __name__ == "__main__":
    dataset_name = "intron.30C"
    model_label = "ssVC"
    positions_labels = POSITION_LABELS[dataset_name]
    positions = np.arange(len(positions_labels))
    alphabet = list("ACGU")
    print("Calculating variance components for MAP estimate")

    print("  Loading MAP estimate..")
    data = pd.read_csv(
        f"results/{dataset_name}.{model_label}.landscape.csv", index_col=0
    )

    print("  Calculating summary statistics")
    s = GPmapSummarizer(n_alleles=4, seq_length=8, f=data["f"].values)

    print("    Calculating gamma_AiBi_to_AjBj statistics")
    gamma_i_to_j = s.calc_gamma_AiBi_to_AjBj()
    add_labels(gamma_i_to_j, positions_labels, alphabet)

    print("        Saving gamma_AiBi_to_AjBj statistics")
    fpath = f"results/{dataset_name}.{model_label}.gamma_AiBi_to_AjBj.csv"
    gamma_i_to_j.to_csv(fpath)

    print("    Calculating gamma_ij_D statistics")
    mutations_D = {
        "C21U": {"D": [7], "D_alleles": [(1, 3)]},
        "G21C": {"D": [7], "D_alleles": [(2, 1)]},
        "C2G-G21C": {"D": [0, 7], "D_alleles": [(1, 2), (2, 1)]},
    }
    gamma_ij_D = []
    for label, kwargs in mutations_D.items():
        print(f"      {label}")
        gamma = s.calc_gamma_AiBiAjBj_D(**kwargs)
        gamma["D"] = label
        gamma_ij_D.append(gamma)
    gamma_ij_D = pd.concat(gamma_ij_D)
    add_labels(gamma_ij_D, positions_labels, alphabet)

    print("        Saving gamma_AiBiAjBj_D statistics")
    fpath = f"results/{dataset_name}.{model_label}.gamma_AiBiAjBj_D.csv"
    gamma_ij_D.to_csv(fpath)

    print("Done.")
