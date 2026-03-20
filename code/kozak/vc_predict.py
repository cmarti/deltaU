import numpy as np

from code.utils import load_experimental_data
from gpmap.inference import VCregression


if __name__ == "__main__":
    dataset = "kozak"
    data = load_experimental_data(dataset)
    X_train, y_train, y_var_train, X_test, y_test, y_var_test = data

    lambdas = np.load("results/vcregression.lambdas.npy")
    model = VCregression(seq_length=6, alphabet_type="rna", lambdas=lambdas)
    model.set_data(X=X_train, y=y_train, y_var=y_var_train)

    print("Computing MAP for complete sequence-space")
    inferred = model.predict()
    inferred.to_csv("results/vcregression.map.csv")

    print("Computing posterior variances for test data")
    test_pred = model.predict(X_pred=X_test, calc_variance=True)
    test_pred.to_csv("results/vcregression.test_pred.csv")
