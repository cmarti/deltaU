import numpy as np
import pandas as pd

if __name__ == "__main__":
    np.random.seed(0)
    for temp in [30, 37]:
        print(f'Loading processed data for {temp}C')
        fpath = f"data/processed/intron.{temp}C.csv"
        data = pd.read_csv(fpath, index_col=0)
        
        print('  Splitting into train and test sets')
        idx = np.random.uniform(size=data.shape[0]) < 0.005
        train = data.loc[~idx, :]
        test = data.loc[idx, :]
        
        print('  Storing train and test sets')
        train.to_csv(f"data/processed/intron.{temp}C.train.csv")
        test.to_csv(f"data/processed/intron.{temp}C.test.csv")
    print('Done.')
