from code.plot_utils import DATASETS

import numpy as np
import pandas as pd
from gpmap.datasets import DataSet, list_available_datasets
from gpmap.inference import (
    ConnectednessModelRegression,
    LocalEpistasisRegression,
    MinimumEpistasisInterpolator,
    VCregression,
)
from scipy.stats import pearsonr

if __name__ == "__main__":
    for dataset_name in ['intron.30C'] + DATASETS:
        np.random.seed(0)
        print(f"Calculating r2 curves for {dataset_name} dataset...")
        
        print("  Loading data")
        if dataset_name not in list_available_datasets():
            data = pd.read_csv(f'data/processed/{dataset_name}.csv', index_col=0)
        else:
            data = DataSet(dataset_name).data
        X, y, y_var = data.index.values, data.y.values, data.y_var.values

        models = {
            "MEI": MinimumEpistasisInterpolator(genotypes=X, P=2),
            "VC": VCregression(genotypes=X),
            "CN": ConnectednessModelRegression(genotypes=X),
            "LER": LocalEpistasisRegression(genotypes=X, P=2),
        }

        print("Calculating R2 curves")
        results = []
        for p in np.geomspace(0.01, 0.99, 10):
            n_train = int(p * data.shape[0])
            for i in range(3):
                print(
                    f"  Randomly splitting training/test: iter {i + 1} for p={p}"
                )

                # Split train/test sets
                u = np.random.uniform(size=data.shape[0])
                q = np.percentile(u, q=p * 100)
                train_idx = u < q
                test_idx = ~train_idx

                X_train, y_train, y_var_train = (
                    X[train_idx],
                    y[train_idx],
                    y_var[train_idx],
                )
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
