import numpy as np
import pandas as pd
from gpmap.inference import VCregression

if __name__ == "__main__":
    dataset_label = "intron.30C"

    print(f"Fitting VC Regression model to {dataset_label} data")
    data = pd.read_csv(
        f"data/processed/{dataset_label}.train.csv", index_col=0
    )
    X, y, y_var = (data.index.values, data["y"].values, data["y_var"].values)
    seq_length = len(X[0])
    print(f"  Loaded {X.shape[0]} training sequences")

    print("  Learning interaction strenghts a_ij")
    model = VCregression(seq_length=seq_length, alphabet_type="dna")
    model.fit(X, y, y_var=y_var)

    print("  Saving interaction strenghts a_ij")
    fpath = f"results/{dataset_label}.vc.lambdas.npy"
    np.save(fpath, model.lambdas)

    # print("  Saving predicted and observed distance-correlation function")
    # corrs_df = model.get_empirical_pred_correlations_df()
    # corrs_df["seq"] = corrs_df.index
    # corrs_df.to_csv(f"results/{dataset_label}.corrs.csv", index=False)
    # print("Done.")
