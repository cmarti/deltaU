import numpy as np
import pandas as pd
from gpmap.aligner import SitesVCKernelAligner
from gpmap.inference import LocalEpistasisRegression, SitesVCregression

if __name__ == "__main__":
    np.random.seed(0)

    print("Initializing LER model")
    model = LocalEpistasisRegression(seq_length=8, alphabet_type="dna", P=2)
    n_Us = len(model.Us)

    # Initialize a_values to high value background interaction strength
    Us_idx = dict(zip(model.Us, np.arange(n_Us)))
    a_values = np.full(n_Us, 5e-2)

    # Add base pair stacking interactions
    pairs = [
        (0, 1),
        (0, 6),
        (0, 7),
        (1, 2),
        (1, 5),
        (1, 6),
        (1, 7),
        (2, 3),
        (2, 4),
        (2, 5),
        (2, 6),
        (3, 4),
        (3, 5),
        (4, 5),
        (5, 6),
        (6, 7),
    ]
    for pair in pairs:
        a_values[Us_idx[pair]] = 1e-4

    # Initialize lower order lambda_U
    lambda_U_lower_than_P = np.array([2e2] + [2e3] * model.seq_length)

    # Initialize model parameters
    model.set_lambda_Us(
        a_values=a_values, lambda_U_lower_than_P=lambda_U_lower_than_P
    )

    print("  Saving LER prior interaction strenghts matrix")
    position_labels = np.arange(1, model.seq_length + 1)
    a_df = model.get_a_values(position_labels=position_labels)
    a_matrix = pd.pivot_table(
        a_df, index="site1", columns="site2", values="interaction_strength"
    )
    a_matrix = (
        a_matrix.reindex(position_labels)
        .fillna(0)
        .T.reindex(position_labels)
        .fillna(0)
        .T
    )
    a_matrix = a_matrix + a_matrix.T
    a_matrix.to_csv("results/simulations.ler.prior_a.csv")

    print("  Saving LER prior correlations")
    prior_cov = model.aligner.predict(model.get_params())
    prior_cor = prior_cov / prior_cov[0]
    # params = model.aligner.fit(prior_cor, np.ones_like(prior_cor))
    # print(params)
    Us = model.all_Us.astype(int).astype(str)
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
    prior_cor_df.to_csv(
        "results/simulations.ler.prior_correlations.csv", index=False
    )

    lambda_U = pd.DataFrame({"U": sites, "k": d, "lambda_U": model.lambdas})
    lambda_U.to_csv("results/simulations.ler.lambda_U.csv", index=False)

    print("  Sampling f from the LER prior")
    K_sqrt = model.K
    K_sqrt.set_lambdas(np.sqrt(model.K.lambdas))
    f = K_sqrt @ np.random.normal(size=K_sqrt.shape[0])
    f_std = np.std(f)

    y_sd = np.full_like(f, 0.2)
    y_var = np.square(y_sd)
    y = np.random.normal(f, scale=y_sd)
    data = pd.DataFrame(
        {"f": f, "y": y, "y_var": y_var}, index=model.genotypes
    )
    data.to_csv("data/processed/simulations.ler.csv")

    #################################################################

    print("Initializing ssVC model")
    model = SitesVCregression(seq_length=8, alphabet_type="dna")

    print("  Re-scaling variance components for ssVC prior")
    lambda_k = np.append(np.array([1, 1]), np.geomspace(1, 10, 7))
    k_factor = np.array([lambda_k[k] for k in lambda_U["k"]])
    lambda_U["lambda_U"] = lambda_U["lambda_U"] * k_factor
    lambda_U.to_csv("results/simulations.ssVC.lambda_U.csv", index=False)
    
    print("  Sampling f from the ssVC prior")
    model.set_lambdas(lambda_U["lambda_U"].values)
    K_sqrt = model.K
    K_sqrt.set_lambdas(np.sqrt(model.K.lambdas))
    f = K_sqrt @ np.random.normal(size=K_sqrt.shape[0])
    f *= f_std / np.std(f) # Re-scaling to match the variance of the LER prior

    y_sd = np.full_like(f, 0.2)
    y_var = np.square(y_sd)
    y = np.random.normal(f, scale=y_sd)
    data = pd.DataFrame(
        {"f": f, "y": y, "y_var": y_var}, index=model.genotypes
    )
    data.to_csv("data/processed/simulations.ssVC.csv")
    
    print("  Saving ssVC prior correlations")
    aligner = SitesVCKernelAligner(n_alleles=4, seq_length=8)
    prior_cov = aligner.predict(model.lambdas)
    prior_cor = prior_cov / prior_cov[0]
    prior_cor_df = pd.DataFrame(
        {
            "d": d,
            "cov": prior_cov,
            "cor": prior_cor,
            "d_jittered": np.random.normal(d, scale=0.05),
            "seq": sites,
        },
    )
    prior_cor_df.to_csv(
        "results/simulations.ssVC.prior_correlations.csv", index=False
    )

    print("Done.")
