import numpy as np
import pandas as pd

from scipy.stats import pearsonr
from gpmap.datasets import DataSet
from gpmap.inference import (
    MinimumEpistasisInterpolator,
    ConnectednessModelRegression,
    VCregression,
    LocalEpistasisRegression,
)

if __name__ == "__main__":
    dataset_name = "smn1"
    np.random.seed(0)

    print("Loading data")
    data = pd.read_csv(f"data/processed/{dataset_name}.csv", index_col=0)
    data["y_var"] = data["y_std"] ** 2
    X, y, y_var = data.index.values, data.y.values, data.y_var.values + 50
    print(y.mean(), y.max())
    models = {
        "MEI": MinimumEpistasisInterpolator(
            seq_length=8, alphabet_type="rna", P=2
        ),
        # "VC": VCregression(seq_length=8, alphabet_type="rna", cg_rtol=1e-2),
        "CN": ConnectednessModelRegression(
            seq_length=8, alphabet_type="rna", cg_rtol=1e-2
        ),
        # "LER": LocalEpistasisRegression(seq_length=8, alphabet_type="rna", P=2),
    }

    print("Calculating R2 curves")
    results = []
    for p in np.geomspace(0.01, 0.99, 10):
        n_train = int(p * data.shape[0])
        for _ in range(1):
            train_idx = np.random.choice(
                data.index, size=n_train, replace=False
            )
            train = data.loc[train_idx, :]
            test_idx = ~np.isin(data.index, train_idx)
            test = data.loc[test_idx, :]

            X_train = train.index.values
            y_train = train.y.values
            y_var_train = train.y_var.values
            X_test, y_test = test.index.values, test.y.values

            # Make predictions using the VC regression model
            for label, model in models.items():
                print("Fitting")
                model.fit(X=X_train, y=y_train, y_var=y_var_train)

                print("Predicting")
                y_pred = model.predict()
                f_train_pred = y_pred.loc[X_train, "f"].values
                f_test_pred = y_pred.loc[X_test, "f"].values

                r2_train = pearsonr(f_train_pred, y_train)[0] ** 2  # type: ignore
                rmse_train = np.sqrt(np.mean((y_train - f_train_pred) ** 2))
                r2_test = pearsonr(f_test_pred, y_test)[0] ** 2  # type: ignore
                rmse_test = np.sqrt(np.mean((y_test - f_test_pred) ** 2))
                record = {
                    "p": p,
                    "r2_train": r2_train,
                    "rmse_train": rmse_train,
                    "r2_test": r2_test,
                    "rmse_test": rmse_test,
                    "model": label,
                }
                print(record)
                results.append(record)

    results = pd.DataFrame(results)
    results.to_csv(f"results/{dataset_name}.r2.csv")
