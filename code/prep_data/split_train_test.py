import pandas as pd
import numpy as np


if __name__ == "__main__":
    np.random.seed(0)
    fpath = "data/processed/intron.lfc.csv"
    data = pd.read_csv(fpath, index_col=0)
    idx = np.random.uniform(size=data.shape[0]) < 0.005
    train = data.loc[~idx, :]
    test = data.loc[idx, :]
    train.to_csv("data/processed/intron.train.csv")
    test.to_csv("data/processed/intron.test.csv")
