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
    fpath = "data/raw/intron.csv"
    wt = "AGGTACAT"
    data = pd.read_csv(fpath, index_col=0)
    cols = []
    design = {}
    design2 = {}
    contrasts = {"37C_y": {}, "30C_y": {}}
    for temp in [30, 37]:
        for rep in range(1, 6):
            col = f"Kan-{temp}C_R{rep}"
            cols.append(col)
            design[col] = {f"{temp}C_R{rep}": 1}
            design2[col] = {f"{temp}C_R{rep}": 1}

            col = f"Kan+{temp}C_R{rep}"
            cols.append(col)
            design[col] = {f"{temp}C_R{rep}": 1, f"{temp}C_R{rep}_y": 1}
            design2[col] = {f"{temp}C_R{rep}": 1, f"{temp}C_y": 1}
            contrasts[f"{temp}C_y"][f"{temp}C_R{rep}_y"] = 0.2
    log_norm_factors = np.log(data[cols].sum())
    log_norm_factors -= log_norm_factors[0]

    design = pd.DataFrame(design).fillna(0).T
    design2 = pd.DataFrame(design2).fillna(0).T
    contrasts = pd.DataFrame(contrasts).reindex(design.columns).fillna(0).T

    # subset = data.loc[np.random.uniform(size=data.shape[0]) < 0.005, :]
    # X = block_diag(*[design2] * subset.shape[0])
    # y = subset[cols].values.flatten()
    # print(X.shape, y.shape)

    # alpha = 0.5
    # model = sm.GLM(
    #     y,
    #     X,
    #     family=sm.families.NegativeBinomial(alpha=alpha),
    #     offset=np.hstack([log_norm_factors] * subset.shape[0]),
    # ).fit()
    # print(alpha, model.ll)
    # exit()

    alpha = 0.04
    results = []
    llf = 0
    for seq, counts in tqdm(data[cols].iterrows(), total=data.shape[0]):
        record = {"seq": seq}

        # model = sm.GLM(
        #     counts,
        #     design,
        #     family=sm.families.Poisson(),
        #     offset=log_norm_factors,
        # )
        # res = model.fit()
        # test = res.t_test(contrasts).summary_frame().set_index(contrasts.index)

        # for temp in [30, 37]:
        #     record[f"{temp}C_y"] = test.loc[f"{temp}C_y", "coef"]
        #     record[f"{temp}C_y_var"] = test.loc[f"{temp}C_y", "std err"] ** 2
        #     for rep in np.arange(1, 6):
        #         record[f"{temp}C_R{rep}_y"] = res.params.loc[
        #             f"{temp}C_R{rep}_y"
        #         ]
        #         record[f"{temp}C_R{rep}_y_var"] = (
        #             res.bse.loc[f"{temp}C_R{rep}_y"] ** 2
        #         )

        model = sm.GLM(
            counts,
            design2,
            family=sm.families.NegativeBinomial(alpha=alpha),
            offset=log_norm_factors,
        )
        res = model.fit()
        for temp in [30, 37]:
            record[f"{temp}C_y"] = res.params.loc[f"{temp}C_y"]
            record[f"{temp}C_y_var"] = res.bse.loc[f"{temp}C_y"] ** 2

        record["deviance"] = res.deviance
        record["perc_dev_explained"] = (
            res.null_deviance - res.deviance
        ) / res.null_deviance
        llf += res.llf

        results.append(record)
        # print(record)
        # break
        # exit()
    results = pd.DataFrame(results)
    results.to_csv("data/processed/intron.nb.csv")
