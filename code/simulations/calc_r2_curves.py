from code.models2 import TruncatedModel, evaluate_predictions

import numpy as np
import pandas as pd
from gpmap.inference import (
    ConnectednessModelRegression,
    LocalEpistasisRegression,
    MinimumEpistasisInterpolator,
    VCregression,
)

if __name__ == "__main__":
    np.random.seed(0)
    print("Loading data")
    data = pd.read_csv("data/processed/simulations.csv", index_col=0)
    X, y, y_var = data.index.values, data.y.values, data.y_var.values

    models = {
        "Additive": TruncatedModel(genotypes=X, max_k=1),
        "Pairwise": TruncatedModel(genotypes=X, max_k=2),
        "Threeway": TruncatedModel(genotypes=X, max_k=3),
        "MEI": MinimumEpistasisInterpolator(
            seq_length=8, alphabet_type="rna", P=2
        ),
        "VC": VCregression(seq_length=8, alphabet_type="rna"),
        "CN": ConnectednessModelRegression(seq_length=8, alphabet_type="rna"),
        "LER": LocalEpistasisRegression(
            seq_length=8, alphabet_type="rna", P=2
        ),
    }

    print("Calculating R2 curves")
    results = []
    for p in np.geomspace(0.01, 0.99, 10):
        n_train = int(p * data.shape[0])
        for _ in range(3):
            train_idx = np.random.choice(
                data.index, size=n_train, replace=False
            )
            train = data.loc[train_idx, :]
            test_idx = ~np.isin(data.index, train_idx)
            test = data.loc[test_idx, :]

            X_train = train.index.values
            f_train = train.f.values
            y_train = train.y.values
            y_var_train = train.y_var.values
            X_test, f_test = test.index.values, test.f.values

            for label, model in models.items():
                model.fit(X=X_train, y=y_train, y_var=y_var_train)
                y_pred = model.predict()

                f_train_pred = y_pred.loc[X_train, "f"].values
                f_test_pred = y_pred.loc[X_test, "f"].values

                record = evaluate_predictions(
                    y_pred, X_train, X_test, y_train, f_test, label=label, p=p
                )
                results.append(record)

    results = pd.DataFrame(results)
    results.to_csv("results/simulations.r2.csv")
