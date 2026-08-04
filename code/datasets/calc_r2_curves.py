import json
import os
from code.models import TruncatedModel, evaluate_predictions
from code.plot_utils import DATASETS

import pandas as pd
import numpy as np
from gpmap.datasets import DataSet, list_available_datasets
from gpmap.inference import (
    ConnectednessModelRegression,
    LocalEpistasisRegression,
    MinimumEpistasisInterpolator,
    SitesVCregression,
    VCregression
)

if __name__ == "__main__":
    for dataset_name in DATASETS:
        print(f'Evaluating dataset {dataset_name}')
        splits = pd.read_csv(f"data/processed/splits/{dataset_name}.splits.csv", index_col=0)
        if dataset_name not in list_available_datasets():
            data = pd.read_csv(f'data/processed/{dataset_name}.csv', index_col=0)
        else:
            data = DataSet(dataset_name).data
        genotypes = data.index.values
        
        # splits = splits.iloc[::-1, :].iloc[:3, :]
        # splits = splits.iloc[:9, :]
        for i, p in zip(splits['i'], splits['p']):
            print(f"  Loading data for split {i} with training p={p:2f}")
            train = pd.read_csv(f'data/processed/splits/{dataset_name}.{i}.train.csv', index_col=0)
            test = pd.read_csv(f'data/processed/splits/{dataset_name}.{i}.test.csv', index_col=0)
            X, y, y_var = train.index.values, train.y.values, train.y_var.values
            X_test, y_test = test.index.values, test.y.values

            cg_rtol = 1e-2
            models = {
                # "MEI": MinimumEpistasisInterpolator(genotypes=X, P=2, cg_rtol=cg_rtol),
                # "VC": VCregression(genotypes=X, cg_rtol=cg_rtol),
                # "CN": ConnectednessModelRegression(genotypes=X, cg_rtol=cg_rtol),
                # "SitesVC": SitesVCregression(genotypes=X, cg_rtol=cg_rtol),
                # "LER": LocalEpistasisRegression(genotypes=X, P=2, cg_rtol=cg_rtol),
                # "Additive": TruncatedModel(genotypes=X, max_k=1),
                # "Pairwise": TruncatedModel(genotypes=X, max_k=2),
                "Threeway": TruncatedModel(genotypes=X, max_k=3),
            }
            
            for label, model in models.items():
                
                print(f"  {label} model")
                print("    Fitting")
                if label == 'MEI':
                    model.fit(X=X, y=y)
                elif label in ['Additive', 'Pairwise', 'Threeway']:
                    # if label == 'Threeway':
                    #     model.optimize_regularization_k3(X=X, y=y, y_var=y_var)
                    model.fit(X=X, y=y)
                else:
                    model.fit(X=X, y=y)
                    model.set_data(X=X, y=y, y_var=y_var) # Set like this to test predictions in SMN1, but not quite working
                    # model.fit(X=X, y=y, y_var=y_var)

                print("    Predicting")
                if label == 'MEI' and dataset_name == 'smn1':
                    pred = model.predict(X_test)
                else:
                    pred = model.predict()
                
                print("    Evaluating")
                record = evaluate_predictions(pred, X, X_test, y, y_test, label, p)
                
                print('    Saving results')
                out_dir = "data/processed/splits"
                print(record)
                os.makedirs(out_dir, exist_ok=True)
                with open(os.path.join(out_dir, f"{dataset_name}.{i}.{label}.json"), "w") as out:
                    json.dump(record, out, indent=2)
