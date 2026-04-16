import numpy as np
import pandas as pd
from gpmap.inference import LocalEpistasisRegression

if __name__ == "__main__":
    for dataset_name in ["smn1", "dmsc", "gb1", "fyn-sh3", "intron.30C"]:
        np.random.seed(0)
        
        print(f"Loading data for {dataset_name}...")
        data = pd.read_csv(
            f"data/processed/{dataset_name}.train.csv", index_col=0
        )
        X, y, y_var = (
            data.index.values,
            data["y"].values,
            data["y_var"].values,
        )
        seq_length = len(X[0])
        print(f"  Loaded {X.shape[0]} training sequences")

        print("Learning interaction strenghts...")
        model = LocalEpistasisRegression(
            seq_length=seq_length, alphabet_type="dna", P=2
        )
        model.fit(X, y)

        print("  Saving correlations under the inferred prior...")
        corrs_df = model.get_empirical_pred_correlations_df()
        corrs_df["seq"] = corrs_df.index
        corrs_df.to_csv(f"results/{dataset_name}.corrs.csv", index=False)
        
        print("  Saving interaction strenghts a_ij")
        fpath = f"results/{dataset_name}.ler.a.npy"
        np.save(fpath, model.a_values)

        print("  Saving interaction lambda_i and lambda_0")
        fpath = f"results/{dataset_name}.ler.lambda_U.npy"
        np.save(fpath, model.lambda_U_lower_than_P)

        print("  Saving interaction strenghts a_ij")
        position_labels = np.arange(1, model.seq_length + 1)
        a_values = model.get_a_values(position_labels=position_labels)
        a_values = pd.pivot_table(
            a_values,
            index="site1",
            columns="site2",
            values="interaction_strength",
        )
        a_values = (
            a_values.reindex(position_labels)
            .fillna(0)
            .T.reindex(position_labels)
            .fillna(0)
            .T
        )
        a_values = a_values + a_values.T
        a_values.to_csv(
            f"results/{dataset_name}.inferred_interaction_strength.csv"
        )
