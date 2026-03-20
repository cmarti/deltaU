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
    dataset_name = "dmsc"
    np.random.seed(0)

    print("Loading data")
    data = DataSet(dataset_name).data
    X, y, y_var = data.index.values, data.y.values, data.y_var.values
    
    # # Normalize measurments to SD=1
    # std = y.std()
    # y = y / std
    # y_var = y_var / std ** 2
    seq_length = len(X[0])

    models = {
        "MEI": MinimumEpistasisInterpolator(
            seq_length=seq_length, alphabet_type="rna", P=2
        ),
        "VC": VCregression(seq_length=seq_length, alphabet_type="rna"),
        "CN": ConnectednessModelRegression(
            seq_length=seq_length, alphabet_type="rna"
        ),
        "LER": LocalEpistasisRegression(
            seq_length=seq_length, alphabet_type="rna", P=2
        ),
    }
    
    # print('Fitting models')
    # for label, model in models.items():
    #     print(f"  Fitting {label} model")
    #     model.fit(X=X, y=y, y_var=y_var)

    print("Calculating R2 curves")
    results = []
    for p in np.geomspace(0.01, 0.99, 10):
        n_train = int(p * data.shape[0])
        for i in range(3):
            print(f"  Randomly splitting training/test: iter {i+1} for p={p}")
            
            # Split train/test sets
            u = np.random.uniform(size=data.shape[0])
            q = np.percentile(u, q=p * 100)
            train_idx = u < q
            test_idx = ~train_idx

            X_train, y_train, y_var_train = X[train_idx], y[train_idx], y_var[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            # Make predictions using the VC regression model
            for label, model in models.items():
                print("Fitting")
                model.fit(X=X_train, y=y_train, y_var=y_var_train)
                # model.set_data(X=X_train, y=y_train, y_var=y_var_train)

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
