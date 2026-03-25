import numpy as np
import pandas as pd
from gpmap.datasets import DataSet, list_available_datasets

if __name__ == "__main__":
    for dataset_name in ["smn1", "dmsc", "gb1", 'fyn-sh3']:
        np.random.seed(0)
        p = 0.2

        print(f"Loading data for {dataset_name} dataset...")
        if dataset_name not in list_available_datasets():
            data = pd.read_csv(f'data/processed/{dataset_name}.csv', index_col=0)
        else:
            data = DataSet(dataset_name).data

        print(f"  Splitting intro train/test sets ({p * 100}% test)...")
        idx = np.random.uniform(size=data.shape[0]) < p
        train = data.loc[~idx, :]
        test = data.loc[idx, :]

        print("  Saving train/test sets ...")
        train.to_csv(f"data/processed/{dataset_name}.train.csv")
        test.to_csv(f"data/processed/{dataset_name}.test.csv")
    print("Done.")
