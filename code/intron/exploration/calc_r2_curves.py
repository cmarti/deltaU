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


def generate_cv_curve_data(data):
    for p in np.geomspace(0.05, 0.95, 2)[::-1]:
        for i in range(3):
            print(
                f"  Randomly splitting training/test: iter {i + 1} for p={p}"
            )

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
            yield(p, i, X_train, y_train, y_var_train, X_test, y_test)


def evaluate_predictions(y_pred, X_train, X_test, y_train, y_test):
    f_train_pred = y_pred.loc[X_train, "f"].values
    f_test_pred = y_pred.loc[X_test, "f"].values

    r2_train = pearsonr(f_train_pred, y_train)[0] ** 2  # type: ignore
    rmse_train = np.sqrt(np.mean((y_train - f_train_pred) ** 2))
    mae_train = np.mean(np.abs(y_train - f_train_pred))
    
    r2_test = pearsonr(f_test_pred, y_test)[0] ** 2  # type: ignore
    rmse_test = np.sqrt(np.mean((y_test - f_test_pred) ** 2))
    mae_test = np.mean(np.abs(y_test - f_test_pred))
    record = {
        "p": p,
        "r2_train": r2_train,
        "rmse_train": rmse_train,
        "mae_train": mae_train,
        "r2_test": r2_test,
        "rmse_test": rmse_test,
        "mae_test": mae_test,
        "model": label,
    }
    return(record)


if __name__ == "__main__":
    for dataset_name in ['smn1']: # DATASETS[:1]:
        np.random.seed(0)
        print(f"Calculating r2 curves for {dataset_name} dataset...")
        
        print("  Loading data")
        if dataset_name not in list_available_datasets():
            data = pd.read_csv(f'data/processed/{dataset_name}.csv', index_col=0)
        else:
            data = DataSet(dataset_name).data
        X, y, y_var = data.index.values, data.y.values, data.y_var.values
        y_var += 0.1 * y.var()
        # print(y.var(), y_var.mean())
        # continue

        models = {
            # "MEI": MinimumEpistasisInterpolator(genotypes=X, P=2),
            # "VC": VCregression(genotypes=X),
            "CN": ConnectednessModelRegression(genotypes=X),
            # "LER": LocalEpistasisRegression(genotypes=X, P=2),
        }
        
        # print("Fitting models")
        # for label, model in models.items():
        #     print(f"  {label}")
        #     model.fit(X=X, y=y, y_var=y_var)

        print("Calculating R2 curves")
        results = []
        for p, i, X_train, y_train, y_var_train, X_test, y_test in generate_cv_curve_data(data):
            
            for label, model in models.items():
                print("Fitting")
                model.fit(X=X_train, y=y_train, y_var=y_var_train)
                # model.set_data(X=X_train, y=y_train, y_var=y_var_train)

                print("Predicting")
                pred = model.predict()
                
                print("Evaluating")
                record = evaluate_predictions(pred, X_train, X_test, y_train, y_test)
                
                print(record)
                results.append(record)

        results = pd.DataFrame(results)
        results.to_csv(f"results/{dataset_name}.r2.csv")
        # exit()
