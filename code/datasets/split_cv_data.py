from code.plot_utils import DATASETS

import numpy as np
import pandas as pd
from gpmap.datasets import DataSet, list_available_datasets


def generate_cv_curve_data(data):
    j = 0
    ps = [0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 0.95]
    for p in ps:
        print(f"    For p={p:.2f}")
        for _ in range(3):
            j += 1
            u = np.random.uniform(size=data.shape[0])
            q = np.percentile(u, q=p * 100)
            train_idx = u < q
            test_idx = ~train_idx

            train = data.loc[train_idx, :]
            test = data.loc[test_idx, :]
            yield(p, j, train, test)


if __name__ == "__main__":
    for dataset_name in DATASETS:
        np.random.seed(0)
        print(f"Splitting data for r2 curves for {dataset_name} dataset...")
        
        print("  Loading data")
        if dataset_name not in list_available_datasets():
            data = pd.read_csv(f'data/processed/{dataset_name}.csv', index_col=0)
        else:
            data = DataSet(dataset_name).data
        sd = data['y'].std()
        data['y'] /= sd
        data['y_var'] /= sd ** 2

        print("  Making splits")
        results = []
        for p, i, train, test in generate_cv_curve_data(data):
            results.append({'i': i, 'p': p})
            train.to_csv(f"data/processed/splits/{dataset_name}.{i}.train.csv")
            test.to_csv(f"data/processed/splits/{dataset_name}.{i}.test.csv")
        results = pd.DataFrame(results)
        results.to_csv(f"data/processed/splits/{dataset_name}.splits.csv")

    print('Done.')