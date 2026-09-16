from code.models import evaluate_predictions

import numpy as np
import pandas as pd
from gpmap.inference import SitesVCregression

if __name__ == "__main__":
    for label in ["ssVC", "ler"]:
        lambda_U = pd.read_csv(f"results/simulations.{label}.lambda_U.csv")
        model = SitesVCregression(
            seq_length=8,
            alphabet_type="rna",
            lambdas=lambda_U["lambda_U"].values,
        )

        data = pd.read_csv(
            f"data/processed/simulations.{label}.csv", index_col=0
        )
        X, y, y_var = data.index.values, data.y.values, data.y_var.values

        print("Calculating R2 curves")
        results = []
        np.random.seed(0)
        for p in np.geomspace(0.01, 0.99, 10):
            print(f"  Using {p * 100:.1f}% of data for training...")
            n_train = int(p * data.shape[0])
            for _ in range(3):
                train_idx = np.random.choice(
                    data.index, size=n_train, replace=False
                )
                train = data.loc[train_idx, :]
                test_idx = ~np.isin(data.index, train_idx)
                test = data.loc[test_idx, :]

                X_train = train.index.values
                f_train = train.f.values
                y_train = train.y.values
                y_var_train = train.y_var.values
                X_test, f_test = test.index.values, test.f.values

                model.set_data(X=X_train, y=y_train, y_var=y_var_train)
                y_pred = model.predict()

                f_train_pred = y_pred.loc[X_train, "f"].values
                f_test_pred = y_pred.loc[X_test, "f"].values

                record = evaluate_predictions(
                    y_pred, X_train, X_test, y_train, f_test, label=label, p=p
                )
                print(record)
                results.append(record)

        results = pd.DataFrame(results)
        results.to_csv(f"results/simulations.{label}.true_lambdas.r2.csv")
