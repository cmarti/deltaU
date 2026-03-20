import pandas as pd
import numpy as np

from gpmap.datasets import DataSet

if __name__ == "__main__":
    dataset_name = "dmsc"
    np.random.seed(0)
    p = 0.2

    print(f"Loading data for {dataset_name} dataset...")
    data = DataSet(dataset_name).data

    print("Splitting intro train/test sets ({}% test)...".format(p * 100))
    idx = np.random.uniform(size=data.shape[0]) < p
    train = data.loc[~idx, :]
    test = data.loc[idx, :]

    print("Saving train/test sets ...")
    train.to_csv(f"data/processed/{dataset_name}.train.csv")
    test.to_csv(f"data/processed/{dataset_name}.test.csv")
    print("Done.")
