import numpy as np
import pandas as pd
from gpmap.inference import LocalEpistasisRegression

if __name__ == "__main__":
    print('Loading data...')
    data = pd.read_csv("data/processed/simulations.train.csv", index_col=0)
    X, y, y_var = (data.index.values, data["y"].values, data["y_var"].values)
    print(f'  Loaded {X.shape[0]} training sequences')
    
    test = pd.read_csv("data/processed/simulations.test.csv", index_col=0)
    X_test = test.index.values
    print(f'  Loaded {X_test.shape[0]} test sequences')

    print("Learning interaction strenghts...")
    model = LocalEpistasisRegression(seq_length=8, alphabet_type="dna", P=2)
    model.fit(X, y, y_var=y_var)

    print("Making predictions for the test set under the inferred prior...")
    pred = model.predict(X_test)
    test = test.join(pred, rsuffix='pred')
    
    print("Saving correlations under the inferred prior...")
    corrs_df = model.get_empirical_pred_correlations_df()
    corrs_df['seq'] = corrs_df.index
    corrs_df.to_csv('results/simulations.corrs.csv', index=False)
    
    print("Saving interaction strenghts a_ij")
    position_labels = np.arange(1, model.seq_length + 1)
    a_values = model.get_a_values(position_labels=position_labels)
    a_values = pd.pivot_table(a_values, index='site1', columns='site2', values='interaction_strength')
    a_values = a_values.reindex(position_labels).fillna(0).T.reindex(position_labels).fillna(0).T
    a_values = (a_values + a_values.T)
    a_values.to_csv('results/simulations.inferred_interaction_strength.csv')
    
    print("Saving model predictions on test set...")
    test.to_csv('results/simulations.pred.csv')
    print('Done.')