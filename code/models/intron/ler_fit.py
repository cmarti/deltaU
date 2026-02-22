import numpy as np
import pandas as pd

from gpmap.inference import LocalEpistasisRegression


if __name__ == "__main__":
    print("Loading processed data")
    data = pd.read_csv("data/processed/intron.train.csv", index_col=0).dropna()
    # data = data.set_index("seq")
    print(data)
    X, y = data.index.values, data["30C_y"].values
    # y = y - y.mean()
    y_var = (
        None if "30C_y_var" not in data.columns else data["30C_y_var"].values
    )

    print("Learning interaction strenghts a_ij")
    model = LocalEpistasisRegression(seq_length=8, alphabet_type="dna", P=2)
    model.fit(X, y, y_var=y_var)

    print("Saving interaction strenghts a_ij")
    fpath = "results/intron.ler.a.npy"
    np.save(fpath, model.a_values)

    print("Saving interaction lambda_U")
    fpath = "results/intron.ler.lambda_U.npy"
    np.save(fpath, model.lambda_U_lower_than_P)

    print("Saving interaction strenghts matrix")
    m = np.zeros((8, 8))
    for (i, j), v in zip(model.Us, model.a_values):
        m[i, j] = 1 / v
        m[j, i] = 1 / v
    positions = [2, 3, 4, 5, 18, 19, 20, 21]
    m = pd.DataFrame(m, index=positions, columns=positions)
    m.to_csv("results/intron.interaction_strength.csv")

    print("Saving predicted and observed distance-correlation function")
    sites = np.array(
        [
            "".join(x)
            for x in np.array(model.aligner.U_sites).astype(int).astype(str)
        ]
    )
    d = [x.count("1") for x in sites]
    corrs = model.aligner.covs / model.aligner.covs[0]
    params = np.append(model.lambda_U_lower_than_P, model.a_values)
    pred = model.aligner.predict(params)
    nodes_df = pd.DataFrame(
        {
            "d": d,
            "n": model.aligner.ns.astype(int),
            "corr": corrs,
            "pred": pred / pred[0],
            "dj": np.random.normal(d, scale=0.05),
            "seq": sites,
        },
    )
    nodes_df.to_csv("results/intron.corrs.csv", index=False)
