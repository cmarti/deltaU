from code.models2 import TruncatedModel

import pandas as pd
from gpmap.inference import (
    ConnectednessModelRegression,
    LocalEpistasisRegression,
    MinimumEpistasisInterpolator,
    VCregression,
)

if __name__ == "__main__":
    dataset_label = 'intron.30C'
    
    print(f"Predicting using all models on {dataset_label} data")
    print('  Loading data...')
    data = pd.read_csv(f"data/processed/{dataset_label}.train.csv", index_col=0).dropna()
    X, y, y_var = data.index.values, data["y"].values, data["y_var"].values

    models = {
            "Additive": TruncatedModel(genotypes=X, max_k=1),
            "Pairwise": TruncatedModel(genotypes=X, max_k=2),
            "MEI": MinimumEpistasisInterpolator(genotypes=X, P=2),
            "VC": VCregression(genotypes=X),
            "CN": ConnectednessModelRegression(genotypes=X),
            "LER": LocalEpistasisRegression(genotypes=X, P=2),
        }
        
    pred = {}
    for label, model in models.items():
        print(f"  {label} model")
        print("    Fitting")
        if label == 'MEI':
            model.fit(X=X, y=y)
        elif label in ['Additive', 'Pairwise']:
            model.fit(X=X, y=y, y_var=y_var)
        else:
            model.fit(X=X, y=y)
            model.set_data(X=X, y=y, y_var=y_var)

        print("    Predicting")
        pred[label] = model.predict()['f']
    
    pred = pd.DataFrame(pred)

    print('  Saving predictions...')
    pred.to_csv(f'results/{dataset_label}.models_predictions.csv')
    print('Done.')