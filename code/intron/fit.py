import numpy as np
import pandas as pd

from gpmap.inference import LocalEpistasisRegression


if __name__ == "__main__":
    dataset_label = "intron.37C"

    print(f"Fitting Local Epistasis Regression model to {dataset_label} data")
    position_labels = np.array([2, 3, 4, 5, 18, 19, 20, 21])
    data = pd.read_csv(
        f"data/processed/{dataset_label}.train.csv", index_col=0
    )
    X, y, y_var = (data.index.values, data["y"].values, data["y_var"].values)
    seq_length = len(X[0])
    print(f"  Loaded {X.shape[0]} training sequences")

    print("  Learning interaction strenghts a_ij")
    model = LocalEpistasisRegression(
        seq_length=seq_length, alphabet_type="dna", P=2
    )
    model.fit(X, y, y_var=y_var)

    print("  Saving interaction strenghts a_ij")
    fpath = f"results/{dataset_label}.ler.a.npy"
    np.save(fpath, model.a_values)

    print("  Saving interaction lambda_i and lambda_0")
    fpath = f"results/{dataset_label}.ler.lambda_U.npy"
    np.save(fpath, model.lambda_U_lower_than_P)

    print("  Saving a_ij matrix for plotting")
    a_values = model.get_a_values(position_labels=position_labels)
    a_values = pd.pivot_table(
        a_values, index="site1", columns="site2", values="interaction_strength"
    )
    a_values = (
        a_values.reindex(position_labels)
        .fillna(0)
        .T.reindex(position_labels)
        .fillna(0)
        .T
    )
    a_values = a_values + a_values.T
    a_values.to_csv(f"results/{dataset_label}.interaction_strength.csv")

    print("  Saving predicted and observed distance-correlation function")
    corrs_df = model.get_empirical_pred_correlations_df()
    corrs_df["seq"] = corrs_df.index
    corrs_df.to_csv(f"results/{dataset_label}.corrs.csv", index=False)
    print("Done.")
