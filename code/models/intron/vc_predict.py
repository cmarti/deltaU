import numpy as np
import pandas as pd

from gpmap.inference import VCregression


if __name__ == "__main__":
    data = pd.read_csv("data/processed/intron.train.csv", index_col=0).dropna()
    X, y, y_var = (
        data.index.values,
        data["30C_y"].values,
        data["30C_y_var"].values,
    )

    lambdas = np.load("results/intron.vcregression.lambdas.npy")
    model = VCregression(seq_length=8, alphabet_type="dna", lambdas=lambdas)
    model.set_data(X, y, y_var=y_var)
    pred = model.predict()
    pred.to_csv('results/intron.vcregression.landscape.csv')
