import numpy as np
import pandas as pd

from collections import defaultdict
from itertools import product
from scipy.special import logsumexp
from gpmap.linop import VUProjectionOperator, ProjectionOperator

if __name__ == "__main__":
    np.random.seed(0)
    a, l = 4, 10

    data = pd.read_csv(
        "data/processed/yiyun_simulated_landscape.csv", index_col=0
    )
    f = data["logp"].values
    windows = [set(range(i, i + 4)) for i in range(l- 4 + 1)]

    windows_data = {
        tuple(w): defaultdict(lambda: np.zeros(a**l)) for w in windows
    }

    # Project function into components depending on wether U-sites are
    # in the target sequence, outside, or across for different windows.
    for U in product([False, True], repeat=l):
        U = set(np.where(U)[0])
        PU = VUProjectionOperator(a, l, U)
        f_U = PU @ f

        for window in windows:
            window_key = tuple(window)

            if U.issubset(window):
                windows_data[window_key]["target_logp"] += f_U
            elif U.isdisjoint(window):
                windows_data[window_key]["across_logp"] += f_U
            else:
                windows_data[window_key]["background_logp"] += f_U

    # Define projection operator into the higher order subspace k>2
    P_ho = ProjectionOperator(a, 4, lambdas=[0, 0, 0, 1, 1])

    for window in windows:
        positions = sorted(window)
        wdata = pd.DataFrame(windows_data[tuple(window)], index=data.index)

        # Compute the true logp by summing the 3 components
        wdata["full_logp"] = wdata.sum(1)
        wdata["p"] = np.exp(wdata["full_logp"])

        # Check that we recover the true logp by comparing with the original values
        assert np.allclose(wdata["full_logp"], f)

        # Compute baseline using interactions within the background
        # and across target-background sites
        wdata["across+background_logp"] = (
            wdata["across_logp"] + wdata["background_logp"]
        )
        wdata["across+background_p"] = np.exp(
            wdata["across+background_logp"]
            - logsumexp(wdata["across+background_logp"])
        )

        # Define target sequence
        wdata["target_seq"] = [
            "".join([x[p] for p in positions]) for x in wdata.index
        ]

        # Summing/averaging over the target sequence
        wmarginal = wdata.groupby("target_seq").agg(
            {
                "full_logp": "mean",
                "target_logp": "mean",
                "p": "sum",
                "across+background_p": "sum",
            }
        )

        # This is what we aim to infer (no interactions with outside positions)
        wmarginal["full_logp"] -= logsumexp(wmarginal["full_logp"])

        # Verify it matches the target logp
        wmarginal["target_logp"] -= logsumexp(wmarginal["target_logp"])
        assert np.allclose(wmarginal["full_logp"], wmarginal["target_logp"])

        # This is the true marginal distribution: should maximize the likelihood
        wmarginal["true_marginal_logp"] = np.log(wmarginal["p"])

        # Baseline calculation from interactions outside and across
        wmarginal["baseline_logp_all_orders"] = np.log(
            wmarginal["across+background_p"]
        )

        # Removing low-order component in the baseline
        wmarginal["baseline_logp_high_orders"] = (
            P_ho @ wmarginal["baseline_logp_all_orders"]
        )

        # Saving dataframe for each window
        label = "".join([str(x) for x in positions])
        fpath = f"data/processed/yiyun_simulated_landscape.window_{label}.csv"
        wmarginal.to_csv(fpath)
