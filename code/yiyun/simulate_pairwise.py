import numpy as np
import pandas as pd

from scipy.special import logsumexp
from gpmap.summary import GPmapSummarizer
from gpmap.linop import ProjectionOperator

if __name__ == "__main__":
    np.random.seed(0)
    l, P = 10, 3
    alphabet = list("ACGT")
    alpha = len(alphabet)
    data = pd.read_csv(
        "data/processed/yiyun_simulated_landscape.csv", index_col=0
    )
    genotypes = data.index.values
    f = data["logp"].values

    # Extract pairwise component
    lambdas = np.zeros(l + 1)
    lambdas[:3] = 1
    P = ProjectionOperator(alpha, l, lambdas=lambdas)
    f_pw = P @ f
    s = GPmapSummarizer(alpha, l, f_pw)
    vc = s.calc_V_k_variance_components()

    logp = f_pw - logsumexp(f_pw)
    p = np.exp(logp)
    sample = np.random.choice(genotypes, size=10000, replace=True, p=p)
    seqs, counts = np.unique(sample, return_counts=True)

    data = pd.DataFrame({"phi": -f_pw, "logp": logp}, index=genotypes)
    data["counts"] = 0

    data.loc[seqs, "counts"] = counts
    data.to_csv("data/processed/yiyun_simulated_landscape_pairwise.csv")
