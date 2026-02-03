import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from gpmap.randwalk import WMWalk
from gpmap.space import SequenceSpace


if __name__ == "__main__":
    print("Loading VC regression MAP")
    data = pd.read_csv("results/vcregression.map.csv", index_col=0)
    X, y = data.index.values, 1 - data.f.values
    print(y.mean(), y.max())

    print("Calculating visualization for VC regression MAP")
    space = SequenceSpace(X, y)

    seqs = ["CACGCU", "AAGAUC", "ACCAUU", "ACAAGA"]
    n = len(seqs)

    fig, subplots = plt.subplots(n, 1, figsize=(3, 2 * 3), sharex=True)
    for seq, axes in zip(seqs, subplots):
        m = space.get_single_mutant_matrix(sequence=seq)
        m = m - m.mean(1).values.reshape((6, 1))
        sns.heatmap(
            m.T,
            ax=axes,
            cmap="coolwarm",
            center=0,
            cbar_kws={"shrink": 1, "label": "Mutational effect"},
        )
        axes.set(
            title=f"{seq} background",
            xlabel="Position",
            ylabel="Allele",
            aspect="equal",
        )
        print(m)
    sns.despine(right=False, top=False)
    fig.tight_layout()
    fig.savefig("figures/mut_effs.png", dpi=300)
