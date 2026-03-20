import pandas as pd
import numpy as np


if __name__ == "__main__":
    fpath = "data/Supp_table_4_kozak_sortseq_and_infection.csv"
    data = pd.read_csv(fpath, index_col=0)
    data["y"] = data["sort_geomean"]
    data["y_var"] = (
        (data["sort_upper_conf"] - data["sort_lower_conf"]) / 4
    ) ** 2
    data = data[["y", "y_var"]].dropna()
    data.index = [x[:-3].replace("T", "U") for x in data.index]

    idx = np.random.uniform(size=data.shape[0]) < 0.05

    train = data.loc[~idx, :]
    test = data.loc[idx, :]

    train.to_csv("processed/kozak.train.csv")
    test.to_csv("processed/kozak.test.csv")
