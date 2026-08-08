from code.plot_utils import POSITION_LABELS

import numpy as np
import pandas as pd

from itertools import product, combinations


def get_seqs_with_mutations(
    positions, alleles, seq_length, alphabet=list("ACGU")
):
    alphabets = [alphabet] * seq_length
    for p, allele in zip(positions, alleles):
        alphabets[p] = [allele]
    seqs = np.array([["".join(x) for x in product(*alphabets)]])
    return seqs


def calc_epistatic_coeff(f, i, j, a_i_1, a_i_2, a_j_1, a_j_2):
    s00 = get_seqs_with_mutations([i, j], [a_i_1, a_j_1], 8)
    s01 = get_seqs_with_mutations([i, j], [a_i_1, a_j_2], 8)
    s10 = get_seqs_with_mutations([i, j], [a_i_2, a_j_1], 8)
    s11 = get_seqs_with_mutations([i, j], [a_i_2, a_j_2], 8)

    f00 = np.array([f[x] for x in s00[0]])
    f01 = np.array([f[x] for x in s01[0]])
    f10 = np.array([f[x] for x in s10[0]])
    f11 = np.array([f[x] for x in s11[0]])

    return f00 + f11 - f01 - f10


if __name__ == "__main__":
    dataset_name = "intron.30C"
    positions_labels = POSITION_LABELS[dataset_name]
    positions = np.arange(len(positions_labels))
    print("Calculating variance components for MAP estimate")

    print("  Loading MAP estimate..")
    data = pd.read_csv(
        f"results/{dataset_name}.ler.landscape.csv", index_col=0
    ).drop_duplicates()
    data.index = [x.replace("T", "U") for x in data.index]
    f = data["f"].to_dict()

    print("  Calculating epistatic coefficients")
    mutations = list(combinations("ACGU", 2))
    epistatic_coeffs = {}
    for i, j in combinations(positions, 2):
        print("    Between positions", i, j)
        for (a_i_1, a_i_2), (a_j_1, a_j_2) in product(mutations, repeat=2):
            label = f"{a_i_1}{positions_labels[i]}{a_i_2}_{a_j_1}{positions_labels[j]}{a_j_2}"
            epistatic_coeffs[label] = calc_epistatic_coeff(
                f, i, j, a_i_1, a_i_2, a_j_1, a_j_2
            )
    epistatic_coeffs = pd.DataFrame(epistatic_coeffs)
    epistatic_coeffs.to_csv(f"results/{dataset_name}.epistatic_coeffs.csv")
    print("Done.")
