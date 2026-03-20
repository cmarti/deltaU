import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from itertools import product
from scipy.linalg import block_diag
from scipy.stats import pearsonr
from tqdm import tqdm
import statsmodels.api as sm

if __name__ == "__main__":
    np.random.seed(0)
    
    print('Loading raw data')
    fpath = "data/raw/intron.csv"
    data = pd.read_csv(fpath, index_col=0)
    data = data[["Log2 fold-change (37ºC)", "Log2 fold-change (30ºC)"]]
    
    print('Storing processed data')
    data.columns = ["37C_y", "30C_y"]
    data.to_csv("data/processed/intron.csv")
    
    print('Done.')
