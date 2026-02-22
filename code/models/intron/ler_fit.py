import numpy as np
import pandas as pd

from gpmap.inference import LocalEpistasisRegression


if __name__ == "__main__":
<<<<<<< HEAD
    position_labels = np.array([2, 3, 4, 5, 18, 19, 20, 21])
    data = pd.read_csv("data/processed/intron.train.csv", index_col=0).dropna()
    X, y, y_var = (
        data.index.values,
        data["30C_y"].values,
        data["30C_y_var"].values + 0.5,
=======
    print("Loading processed data")
    data = pd.read_csv("data/processed/intron.train.csv", index_col=0).dropna()
    # data = data.set_index("seq")
    print(data)
    X, y = data.index.values, data["30C_y"].values
    # y = y - y.mean()
    y_var = (
        None if "30C_y_var" not in data.columns else data["30C_y_var"].values
>>>>>>> 4d39813cc86ec8e0a4e7559944d4a7d5af45d94c
    )
    print(data["30C_y_var"].values.mean())
    print(np.mean((y - y.mean()) ** 2))

    print("Learning interaction strenghts a_ij")
    model = LocalEpistasisRegression(seq_length=8, alphabet_type="dna", P=2)
    model.fit(X, y, y_var=y_var)

    print("Saving interaction strenghts a_ij")
    fpath = "results/intron.ler.a.npy"
    np.save(fpath, model.a_values)

    print("Saving interaction lambda_U")
    fpath = "results/intron.ler.lambda_U.npy"
    np.save(fpath, model.lambda_U_lower_than_P)
<<<<<<< HEAD
    
    a_values = model.get_a_values(position_labels=position_labels)
    a_values = pd.pivot_table(a_values, index='site1', columns='site2', values='interaction_strength')
    a_values = a_values.reindex(position_labels).fillna(0).T.reindex(position_labels).fillna(0).T
    a_values = (a_values + a_values.T)
    a_values.to_csv('results/intron.interaction_strength.csv')

    corrs_df = model.get_empirical_pred_correlations_df()
    corrs_df['seq'] = corrs_df.index
    corrs_df.to_csv('results/intron.corrs.csv', index=False)
    
    

=======

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
>>>>>>> 4d39813cc86ec8e0a4e7559944d4a7d5af45d94c
