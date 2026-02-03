from code.utils import load_experimental_data
from gpmap.inference import MinimumEpistasisInterpolator


if __name__ == "__main__":
    print("Loading data")
    dataset = "kozak"
    data = load_experimental_data(dataset)
    X_train, y_train, _, X_test, _, _ = data

    print("Computing MAP for complete sequence-space")
    model = MinimumEpistasisInterpolator(seq_length=6, alphabet_type="rna", P=2)
    model.fit(X=X_train, y=y_train)
    inferred = model.predict()
    inferred.to_csv("results/mei.map.csv")

    print("Computing posterior variances for test data")
    pred = model.predict(X_pred=X_test, calc_variance=True)
    pred.to_csv("results/mei.test_pred.csv")
