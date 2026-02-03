import pandas as pd
import numpy as np


if __name__ == "__main__":
    fpath = "data/raw/intron.csv"
    cols = [
        "Genotype (N2-N5/N18-N21)",
        "DimSum fitness (30ºC) ",
        "DimSum sigma (30ºC) ",
        "DimSum fitness (37ºC) ",
        "DimSum sigma (37ºC) ",
    ]
    data = pd.read_csv(fpath, index_col=0, usecols=cols)
    d1 = data[cols[1:3]].copy()
    d1.columns = ['y', 'y_sd']
    d1['y_var'] = d1['y_sd'] ** 2
    d1['temp'] = 30
    d2 = data[cols[3:]].copy()
    d2.columns = ['y', 'y_sd']
    d2['y_var'] = d1['y_sd'] ** 2
    d2['temp'] = 37
    data = pd.concat([d1, d2]).dropna()
    data.index = [x.replace("T", "U") for x in data.index]
    data.to_csv("data/processed/intron.csv")
    
    idx = np.random.uniform(size=data.shape[0]) < 0.05
    train = data.loc[~idx, :]
    test = data.loc[idx, :]

    train.to_csv("data/processed/intron.train.csv")
    test.to_csv("data/processed/intron.test.csv")
