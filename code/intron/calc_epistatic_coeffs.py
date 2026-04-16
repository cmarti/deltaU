from code.plot_utils import POSITION_LABELS

import numpy as np
import pandas as pd


def calc_mut_coeff(f, mut, positions_labels):
    seqs = f.index.values
    a1, pos, a2 = mut[0], int(mut[1:-1]), mut[-1]
    seqs_array = np.array([[c for c in x] for x in seqs])

    alleles = seqs_array.copy()
    alleles[:, pos] = a1
    s1 = np.array(["".join(x) for x in alleles])

    alleles = seqs_array.copy()
    alleles[:, pos] = a2
    s2 = np.array(["".join(x) for x in alleles])

    mut_eff = f.reindex(s2).values - f.reindex(s1).values
    label = f"{a1}{positions_labels[pos]}{a2}"
    return label, mut_eff


def calc_epistatic_coeff(f, mut1, mut2, positions_labels):
    seqs = f.index.values
    a11, pos1, a12 = mut1[0], int(mut1[1:-1]), mut1[-1]
    a21, pos2, a22 = mut2[0], int(mut2[1:-1]), mut2[-1]
    seqs_array = np.array([[c for c in x] for x in seqs])

    alleles = seqs_array.copy()
    alleles[:, pos1] = a11
    alleles[:, pos2] = a12
    s11 = np.array(["".join(x) for x in alleles])

    alleles = seqs_array.copy()
    alleles[:, pos1] = a11
    alleles[:, pos2] = a22
    s12 = np.array(["".join(x) for x in alleles])

    alleles = seqs_array.copy()
    alleles[:, pos1] = a21
    alleles[:, pos2] = a12
    s21 = np.array(["".join(x) for x in alleles])

    alleles = seqs_array.copy()
    alleles[:, pos1] = a21
    alleles[:, pos2] = a22
    s22 = np.array(["".join(x) for x in alleles])

    epistatic_coeffs = (
        f.reindex(s22).values
        + f.reindex(s11).values
        - f.reindex(s12).values
        - f.reindex(s21).values
    )
    label = (
        f"{a11}{positions_labels[pos1]}{a12}_{a21}{positions_labels[pos2]}{a22}"
    )
    return label, epistatic_coeffs


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

    results = {}
    print("  Calculating mutational effects")
    mutations = ["G0C", "A0U", "G1C", "C6G", "A1U", "U6A", "G5C", "C6G", 'C7G']
    for mut in mutations:
        print(f"    For mutation {mut}")
        label, values = calc_mut_coeff(data["f"], mut, positions_labels)
        results[label] = values

    print("  Calculating epistatic coefficients")
    mutation_pairs = [["G1C", "C6G"], ["A1U", "U6A"], ["G5C", "C6G"]]
    for mut1, mut2 in mutation_pairs:
        print(f"    For mutations {mut1}-{mut2}")
        label, values = calc_epistatic_coeff(
            data["f"], mut1, mut2, positions_labels
        )
        results[label] = values
    results = pd.DataFrame(results, index=data.index)

    print("Saving computed mutational effects and epistatic coefficients")
    results.to_csv(f"results/{dataset_name}.ler.epistatic_coefficients.csv")

    print("Done.")
