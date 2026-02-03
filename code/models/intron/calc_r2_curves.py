import numpy as np
import pandas as pd

from scipy.stats import pearsonr
from gpmap.inference import (
    VCregression,
    MinimumEpistasisInterpolator,
    ConnectednessModelRegression,
    LocalEpistasisRegression,
)

if __name__ == "__main__":
    data = pd.read_csv("data/processed/intron.csv", index_col=0).dropna()
    data = data[['30C_y', '30C_y_var']]
    data.columns = ['y', 'y_var']
    # data = data.loc[data["temp"] == 37, :]
    # data = data.loc[data['y_var'] < 0.5,:]
    # data['y_var'] += 1
    print(data)

    np.random.seed(0)

    models = {
        # "MEI1": MinimumEpistasisInterpolator(
        #     seq_length=8, alphabet_type="rna", P=1
        # ),
        "MEI2": MinimumEpistasisInterpolator(
            seq_length=8, alphabet_type="rna", P=2
        ),
        # "VC": VCregression(seq_length=8, alphabet_type="rna"),
        # "Connectedness": ConnectednessModelRegression(
        #     seq_length=8, alphabet_type="rna"
        # ),
        # 'LER1': LocalEpistasisRegression(seq_length=8, alphabet_type="rna", P=1),
        "LER2": LocalEpistasisRegression(
            seq_length=8, alphabet_type="rna", P=2
        ),
    }

    results = []
    for p in np.geomspace(0.5, 0.95, 20):
        n_train = int(p * data.shape[0])
        for _ in range(3):
            train_idx = np.random.choice(
                data.index, size=n_train, replace=False
            )
            train = data.loc[train_idx, :]
            test_idx = ~np.isin(data.index, train_idx)
            test = data.loc[test_idx, :]

            X_train, y_train, y_var_train = (
                train.index.values,
                train.y.values,
                train.y_var.values,
            )
            X_test, y_test, y_var_test = (
                test.index.values,
                test.y.values,
                test.y_var.values,
            )

            # Make predictions using the VC regression model
            for label, model in models.items():
                print(f"{label} model")
                print("\tFitting model hyperparameters")
                model.fit(X=X_train, y=y_train, y_var=y_var_train)

                print("\tCalculating predictions on test sequences")
                y_pred = model.predict()
                y_train_pred = y_pred.loc[X_train, 'f'].values
                y_test_pred = y_pred.loc[X_test, 'f'].values

                r2_train = pearsonr(y_train_pred, y_train)[0] ** 2  # type: ignore
                rmse_train = np.sqrt(np.mean((y_train - y_train_pred) ** 2))
                r2_test = pearsonr(y_test_pred, y_test)[0] ** 2  # type: ignore
                rmse_test = np.sqrt(np.mean((y_test - y_test_pred) ** 2))
                record = {
                    "p": p,
                    "r2_train": r2_train,
                    "rmse_train": rmse_train,
                    "r2_test": r2_test,
                    "rmse_test": rmse_test,
                    "model": label,
                }
                print("\t", record)
                results.append(record)

    results = pd.DataFrame(results)
    results.to_csv("results/models.r2.csv")
