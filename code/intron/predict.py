import numpy as np
import pandas as pd

from gpmap.inference import LocalEpistasisRegression


if __name__ == "__main__":
    dataset_label = 'intron.30C'
    
    print(f"Predicting using Local Epistasis Regression model fitted to {dataset_label} data")
    
    print('  Loading data...')
    data = pd.read_csv(f"data/processed/{dataset_label}.train.csv", index_col=0).dropna()
    X, y, y_var = data.index.values, data["y"].values, data["y_var"].values
    # y_var = np.full_like(y, fill_value=0.6)

    print('  Loading model parameters...')
    fpath = f'results/{dataset_label}.ler.a.npy'
    a_values = np.load(fpath)
    print(f"    Loaded a_values: {a_values}")

    fpath = f'results/{dataset_label}.ler.lambda_U.npy'
    lambda_U = np.load(fpath)
    print(f"    Loaded lambda_U: {lambda_U}")

    print('  Making predictions...')
    model = LocalEpistasisRegression(seq_length=8, alphabet_type="dna", P=2,
                                     a_values=a_values, lambda_U_lower_than_P=lambda_U)
    model.set_data(X, y, y_var=y_var)
    pred = model.predict()
    
    print('  Saving predictions...')
    pred.to_csv(f'results/{dataset_label}.ler.landscape.csv')
    print('Done.')