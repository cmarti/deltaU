import json
from code.plot_utils import DATASETS
from os.path import exists

import pandas as pd

if __name__ == "__main__":
    models = ['MEI', 'VC', 'CN', 'LER', 'Additive', 'Pairwise']
    for dataset_name in DATASETS:
        results = []
        splits = pd.read_csv(f"data/processed/splits/{dataset_name}.splits.csv", index_col=0)
        for i, p in zip(splits['i'], splits['p']):
            for label in models:
                fpath = f"data/processed/splits/{dataset_name}.{i}.{label}.json"
                if not exists(fpath):
                    continue
                with open(fpath) as fhand:
                    record = json.load(fhand)
                    record['model'] = label
                    record['p'] = p
                    record['dataset'] = dataset_name
                    results.append(record)
        results = pd.DataFrame(results)
        results.to_csv(f"results/{dataset_name}.r2_curves.csv", index=False)
