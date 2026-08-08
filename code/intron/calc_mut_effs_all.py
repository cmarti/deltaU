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


def calc_mut_eff(f, i, a1, a2):
    s0 = get_seqs_with_mutations([i], [a1], 8)
    s1 = get_seqs_with_mutations([i], [a2], 8)

    f0 = np.array([f[x] for x in s0[0]])
    f1 = np.array([f[x] for x in s1[0]])
    return f1 - f0


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

    print("  Calculating mutational effects")
    mutations = list(combinations("ACGU", 2))
    mut_effs = {}
    for i in positions:
        print("    At position", i)
        for a1, a2 in mutations:
            label = f"{a1}{positions_labels[i]}{a2}"
            mut_effs[label] = calc_mut_eff(f, i, a1, a2)

    mut_effs = pd.DataFrame(mut_effs)
    mut_effs.to_csv(f"results/{dataset_name}.mut_effs.csv")
    print("Done.")
