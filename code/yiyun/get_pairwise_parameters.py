import numpy as np
import pandas as pd

from gaugefixer import AllOrderModel

if __name__ == "__main__":
    np.random.seed(0)
    a, L = 4, 10

    data = pd.read_csv(
        "data/processed/yiyun_simulated_landscape.csv", index_col=0
    )

    model = AllOrderModel(alphabet_name="dna", L=L)
    model.set_landscape(data["logp"])
    theta = model.get_fixed_params(gauge="zero-sum")
    theta = pd.DataFrame({"theta": theta})
    theta["k"] = [len(x[0]) for x in theta.index]
    theta_pw = theta.loc[theta["k"] <= 2, :]
    theta_pw.to_csv("data/processed/yiyun_full_length_pairwise_parameters.csv")
