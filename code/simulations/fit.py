import numpy as np
import pandas as pd
from gpmap.inference import LocalEpistasisRegression, SitesVCregression

if __name__ == "__main__":
    for model_name in ["ssVC", "ler"]:
        print(f"Loading data simulated under {model_name}...")
        data = pd.read_csv(
            f"data/processed/simulations.{model_name}.train.csv", index_col=0
        )
        X, y, y_var = (
            data.index.values,
            data["y"].values,
            data["y_var"].values,
        )
        print(f"  Loaded {X.shape[0]} training sequences")

        test = pd.read_csv(
            f"data/processed/simulations.{model_name}.test.csv", index_col=0
        )
        X_test = test.index.values
        print(f"  Loaded {X_test.shape[0]} test sequences")

        #################################################################

        print("Fitting Local Epistasis Regression model...")
        print("  Learning interaction strenghts...")
        model = LocalEpistasisRegression(
            seq_length=8, alphabet_type="dna", P=2
        )
        model.fit(X, y, y_var=y_var)

        print("  Saving correlations under the inferred prior...")
        corrs_df = model.get_empirical_pred_correlations_df()
        corrs_df["seq"] = corrs_df.index
        corrs_df.to_csv(
            f"results/simulations.{model_name}.corrs.csv", index=False
        )

        print("  Saving inferred lambda_U values...")
        lambda_U = pd.DataFrame(
            {
                "U": corrs_df["seq"].values,
                "k": corrs_df["d"].values,
                "d_jittered": corrs_df["d_jittered"].values,
                "lambda_U": model.lambdas,
            }
        )
        lambda_U.to_csv(
            f"results/simulations.{model_name}.inferred_lambda_U.ler.csv",
            index=False,
        )

        print("  Saving interaction strenghts a_ij...")
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
            f"results/simulations.{model_name}.inferred_interaction_strength.csv"
        )
        
        print("  Making predictions for the test set under the inferred prior...")
        np.random.seed(0)
        X_test = np.random.choice(X_test, size=200, replace=False)
        pred = model.predict(X_test, calc_variance=True)
        test = test.join(pred, rsuffix='pred')
        test.to_csv(f'results/simulations.{model_name}.pred.ler.csv')

        #################################################################

        print(
            "Fitting sites-structured Variance Component Regression model..."
        )
        print("  Learning interaction strenghts...")
        model = SitesVCregression(seq_length=8, alphabet_type="dna")
        model.fit(X, y, y_var=y_var)

        print("  Saving inferred lambda_U values...")
        model.vc_df.to_csv(
            f"results/simulations.{model_name}.inferred_lambda_U.ssVC.csv",
            index=False,
        )

    print("Done.")
