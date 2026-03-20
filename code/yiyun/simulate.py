import numpy as np
import pandas as pd

from itertools import combinations
from scipy.special import logsumexp
from gpmap.inference import LocalEpistasisRegression
from gpmap.summary import GPmapSummarizer

if __name__ == "__main__":
    np.random.seed(0)

    print("Initializing model")
    model = LocalEpistasisRegression(seq_length=9, alphabet_type="dna", P=3)

    # Initialize a_values to high value background interaction strength
    a_values = []
    sel_pos = set([0, 1, 2, 3, 4])
    for U in model.Us:
        if set(U).issubset(sel_pos):
            a_values.append(5e-5)
        else:
            a_values.append(1e16)

    # Initialize lower order lambda_U
    sites = np.array(model.aligner.params_to_log_lambda_U.V)
    # lambda_k = np.array([0, 5e3, 1e2])
    lambda_k = np.array([0, 4e3, 2e2])
    idx = (
        sites[model.aligner.params_to_log_lambda_U.no_U_idx, :]
        .sum(1)
        .astype(int)
    )
    lambda_U_lower_than_P = lambda_k[idx]

    # Initialize model parameters
    model.set_lambda_Us(
        a_values=a_values, lambda_U_lower_than_P=lambda_U_lower_than_P
    )

    print("Saving prior interaction strenghts matrix")
    position_labels = np.arange(1, model.seq_length + 1)
    a_df = model.get_a_values(position_labels=position_labels)
    print(a_df)

    print("Saving prior correlations")
    params = np.append(lambda_U_lower_than_P, a_values)
    prior_cov = model.aligner.predict(params)
    prior_cor = prior_cov / prior_cov[0]
    # params = model.aligner.fit(prior_cor, np.ones_like(prior_cor))
    # print(params)
    Us = (
        np.array(model.aligner.params_to_log_lambda_U.V).astype(int).astype(str)
    )
    sites = np.array(["".join(x) for x in Us])
    d = [x.count("1") for x in sites]
    prior_cor_df = pd.DataFrame(
        {
            "d": d,
            "cov": prior_cov,
            "cor": prior_cor,
            "d_jittered": np.random.normal(d, scale=0.05),
            "seq": sites,
        },
    )
    print(prior_cor_df.groupby("d")[["cov", "cor"]].mean())

    print("Sampling f from the prior")
    K_sqrt = model.K
    K_sqrt.set_lambdas(np.sqrt(model.K.lambdas))
    phi = 6 * K_sqrt @ np.random.normal(size=K_sqrt.shape[0])
    print(phi.mean(), phi.std())

    s = GPmapSummarizer(4, 9, phi)
    vc = s.calc_V_k_variance_components()
    vu = s.calc_V_U_variance_components()
    print(vc)

    logp = -phi - logsumexp(-phi)
    p = np.exp(logp)
    # print(np.quantile(p, [0.1, 0.25, 0.5, 0.75, 0.9]))
    sample = np.random.choice(model.genotypes, size=10000, replace=True, p=p)
    seqs, counts = np.unique(sample, return_counts=True)

    data = pd.DataFrame({"phi": phi, "logp": logp}, index=model.genotypes)
    data["counts"] = 0
    data.loc[seqs, "counts"] = counts
    data.to_csv("data/processed/yiyun_simulated_landscape.csv")
