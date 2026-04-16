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
import json, os


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
    dataset_name = 'fyn-sh3'
    print(f'Evaluating dataset {dataset_name}')
    
    splits = pd.read_csv(f"data/processed/splits/{dataset_name}.splits.csv", index_col=0).set_index('i')['p'].to_dict()
    
    for i in [29]: # splits['i']:
        p = splits[i]
            
        print(f"  Loading data for split {i} with training p={p:2f}")
        if dataset_name not in list_available_datasets():
            data = pd.read_csv(f'data/processed/{dataset_name}.csv', index_col=0)
        else:
            data = DataSet(dataset_name).data
            
        train = pd.read_csv(f'data/processed/splits/{dataset_name}.{i}.train.csv', index_col=0)
        test = pd.read_csv(f'data/processed/splits/{dataset_name}.{i}.test.csv', index_col=0)
        X, y, y_var = train.index.values, train.y.values, train.y_var.values
        X_test, y_test = test.index.values, test.y.values
        # y_var += 0.1 * data.y.values.var()

        models = {
            # "MEI": MinimumEpistasisInterpolator(genotypes=X, P=2),
            # "VC": VCregression(genotypes=X),
            # "CN": ConnectednessModelRegression(genotypes=X),
            "LER": LocalEpistasisRegression(genotypes=X, P=2),
        }
        
        for label, model in models.items():
            print(f"  {label} model")
            print("    Fitting")
            # model.fit(X=X, y=y, y_var=y_var)
            # model.fit(X=X, y=y, y_var=y_var, method='Powell')
            model.fit(X=X, y=y, method='Powell')
            model.set_data(X=X, y=y, y_var=y_var)

            print("    Predicting")
            pred = model.predict()
            
            print("    Evaluating")
            record = evaluate_predictions(pred, X, X_test, y, y_test)
            print(record)
            
            print('    Saving results')
            out_dir = "data/processed/splits"
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, f"{dataset_name}.{i}.{label}.json"), "w") as out:
                json.dump(record, out, indent=2)
