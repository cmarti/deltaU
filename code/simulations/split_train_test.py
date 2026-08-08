import numpy as np
import pandas as pd

if __name__ == "__main__":
    np.random.seed(0)
    p = 0.1
    
    for model in ['ler', 'ssVC']:
        print(f'Loading data for {model} model...')
        fpath = f"data/processed/simulations.{model}.csv"
        data = pd.read_csv(fpath, index_col=0)
        
        print(f'  Splitting into train/test sets ({p*100}% test)...')
        idx = np.random.uniform(size=data.shape[0]) < p
        train = data.loc[~idx, :]
        test = data.loc[idx, :]
        
        print('  Saving train/test sets ...')
        train.to_csv(f"data/processed/simulations.{model}.train.csv")
        test.to_csv(f"data/processed/simulations.{model}.test.csv")
    print('Done.')