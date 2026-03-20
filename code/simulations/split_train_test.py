import pandas as pd
import numpy as np


if __name__ == "__main__":
    np.random.seed(0)
    p = 0.85
    
    print('Loading data ...')
    fpath = "data/processed/simulations.csv"
    data = pd.read_csv(fpath, index_col=0)
    
    print('Splitting intro train/test sets ({}% test)...'.format(p*100))
    idx = np.random.uniform(size=data.shape[0]) < p
    train = data.loc[~idx, :]
    test = data.loc[idx, :]
    
    print('Saving train/test sets ...')
    train.to_csv("data/processed/simulations.train.csv")
    test.to_csv("data/processed/simulations.test.csv")
    print('Done.')