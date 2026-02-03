import pandas as pd
import numpy as np

from code.utils import load_experimental_data
from gpmap.inference import VCregression


if __name__ == "__main__":
    dataset = "kozak"
    X, y, y_var = load_experimental_data(dataset)[:3]

    print("Computing empirical distance-correlation function")
    model = VCregression(seq_length=6, alphabet_type="rna")
    cov, ns = model.calc_covariance_distance(X=X, y=y - y.mean())
    dcor = pd.DataFrame({"d": np.arange(cov.shape[0]), "rho": cov / cov[0]})
    dcor.to_csv("results/vcregression.empirical_distance_correlation.csv")

    print("Estimating variance components")
    model.fit(X, y, y_var)
    np.save("results/vcregression.lambdas.npy", model.lambdas)

    # Save variance components
    print("Saving variance components")
    vc = model.get_variance_components()
    vc.to_csv("results/vcregression.variance_components.csv")
