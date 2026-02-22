import numpy as np
import pandas as pd

from gpmap.inference import LocalEpistasisRegression


if __name__ == "__main__":
    position_labels = np.array([2, 3, 4, 5, 18, 19, 20, 21])
    data = pd.read_csv("data/processed/intron.train.csv", index_col=0).dropna()
    X, y, y_var = (
        data.index.values,
        data["30C_y"].values,
        data["30C_y_var"].values + 0.5,
    )
    print(data["30C_y_var"].values.mean())
    print(np.mean((y - y.mean()) ** 2))

    model = LocalEpistasisRegression(seq_length=8, alphabet_type="dna", P=2)
    model.fit(X, y, y_var=y_var)
    
    fpath = 'results/intron.ler.a.npy'
    np.save(fpath, model.a_values)
    
    fpath = 'results/intron.ler.lambda_U.npy'
    np.save(fpath, model.lambda_U_lower_than_P)
    
    a_values = model.get_a_values(position_labels=position_labels)
    a_values = pd.pivot_table(a_values, index='site1', columns='site2', values='interaction_strength')
    a_values = a_values.reindex(position_labels).fillna(0).T.reindex(position_labels).fillna(0).T
    a_values = (a_values + a_values.T)
    a_values.to_csv('results/intron.interaction_strength.csv')

    corrs_df = model.get_empirical_pred_correlations_df()
    corrs_df['seq'] = corrs_df.index
    corrs_df.to_csv('results/intron.corrs.csv', index=False)
    
    

