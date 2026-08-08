import numpy as np
import pandas as pd
from gpmap.aligner import SitesVCKernelAligner
from gpmap.inference import SitesVCregression

if __name__ == "__main__":
    np.random.seed(0)

    print("Initializing ssVC model")
    model = SitesVCregression(seq_length=8, alphabet_type="dna")
    U = ["".join([str(int(p)) for p in U]) for U in model.Us]
    k = [s.count("1") for s in U]
    log_lambda_k = np.linspace(np.log(1e2), np.log(5e-2), 9)
    
    lambda_U = pd.DataFrame(
        {
            "U": U,
            "k": k,
            'lambda_U': np.exp(log_lambda_k[k] + np.random.normal(0, 1.5, size=len(U)))
        }
    )
    lambda_U.to_csv("results/simulations.ssVC.lambda_U.csv", index=False)
    model.set_lambdas(lambda_U["lambda_U"].values)
    
    print("  Sampling f from the ssVC prior")
    model.set_lambdas(lambda_U["lambda_U"].values)
    K_sqrt = model.K
    K_sqrt.set_lambdas(np.sqrt(model.K.lambdas))
    f = K_sqrt @ np.random.normal(size=K_sqrt.shape[0])
    print(np.std(f))
    # f *= f_std / np.std(f) # Re-scaling to match the variance of the LER prior

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
            "d": k,
            "cov": prior_cov,
            "cor": prior_cor,
            "d_jittered": np.random.normal(k, scale=0.05),
            "seq": U,
        },
    )
    prior_cor_df.to_csv(
        "results/simulations.ssVC.prior_correlations.csv", index=False
    )

    print("Done.")
