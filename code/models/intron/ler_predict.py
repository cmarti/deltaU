import numpy as np
import pandas as pd

from gpmap.inference import LocalEpistasisRegression


if __name__ == "__main__":
    data = pd.read_csv("data/processed/intron.train.csv", index_col=0).dropna()
    X, y, y_var = (
        data.index.values,
        data["30C_y"].values,
        data["30C_y_var"].values + 0.1,
    )

    fpath = 'results/intron.ler.a.npy'
    a_values = np.load(fpath)

    fpath = 'results/intron.ler.lambda_U.npy'
    lambda_U = np.load(fpath)
    lambda_U[1:] = 1e3
    print(lambda_U)

    model = LocalEpistasisRegression(seq_length=8, alphabet_type="dna", P=2,
                                     a_values=a_values, lambda_U_lower_than_P=lambda_U)
    model.set_data(X, y, y_var=y_var)
    pred = model.predict()
    pred.to_csv('results/intron.ler.landscape.csv')
