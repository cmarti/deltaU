import pandas as pd
import numpy as np

from gpmap.inference import VCregression


if __name__ == "__main__":
    data = pd.read_csv("data/processed/intron.train.csv", index_col=0).dropna()
    X, y, y_var = (
        data.index.values,
        data["30C_y"].values,
        data["30C_y_var"].values,
    )

    print("Computing empirical distance-correlation function")
    model = VCregression(seq_length=8, alphabet_type="dna")

    print("Estimating variance components")
    model.fit(X, y, y_var)
    np.save("results/intron.vcregression.lambdas.npy", model.lambdas)

    cov, ns = model.calc_covariance_distance(X=X, y=y)
    d = np.arange(9)
    corrs = cov / cov[0]
    pred = model.kernel_aligner.predict(model.lambdas)
    nodes_df = pd.DataFrame(
        {
            "d": d,
            "n": ns,
            "corr": corrs,
            "pred": pred / pred[0],
            "dj": np.random.normal(d, scale=0.05),
        },
    )
    nodes_df.to_csv('results/intron.vcregression.corrs.csv')

    # Save variance components
    print("Saving variance components")
    vc = model.get_variance_components()
    print(vc)
    vc.to_csv("results/vcregression.intron.variance_components.csv")
