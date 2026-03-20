import pandas as pd
import numpy as np

from gaugefixer import AllOrderModel, PairwiseModel


if __name__ == "__main__":
    dataset_label = "intron.30C"
    wt = "AGGTACAT"

    print(f"Calculating visualization for {dataset_label} dataset")

    print("  Loading inferred landscape...")
    data = pd.read_csv(
        f"results/{dataset_label}.ler.landscape.csv", index_col=0
    )
    model = AllOrderModel(alphabet_name="dna", L=8)
    model.set_landscape(data["f"])
    pw_model = PairwiseModel(alphabet_name="dna", L=8)

    print("  Computing gauge-fixed coefficients")
    pi_uniform = np.full(4, 1 / 4.0)
    dincs = ["AT", "TA", "GC", "CG", "GT", "TG"]
    pi_c = {c: v for c, v in zip("ACGT", np.eye(4))}
    pi_c["N"] = np.full(4, 1 / 4.0)
    theta = {}
    for dinc in dincs:
        print(f"    Gauge defined by {dinc}...")
        pi_lc = [pi_c[dinc[0]]] + [pi_uniform] * 6 + [pi_c[dinc[1]]]
        theta[dinc] = model.get_fixed_params(gauge="hierarchical", pi_lc=pi_lc)

    seqs = [
        "ANNNNNCT",
        "TNNNNNCA",
        "GNNNNNCC",
        "CNNNNNCG",
        "ANNNNNGT",
        "TNNNNNGA",
        "GNNNNNGC",
        "CNNNNNGG",
        "ANNNNCNT",
        "TNNNNCNA",
        "GNNNNCNC",
        "CNNNNCNG",
        "ANNNNGNT",
        "TNNNNGNA",
        "GNNNNGNC",
        "CNNNNGNG",
    ]
    for seq in seqs:
        print(f"    Gauge defined by {seq}...")
        pi_lc = [pi_c[c] for c in seq]
        theta[seq] = model.get_fixed_params(gauge="hierarchical", pi_lc=pi_lc)

    theta = pd.DataFrame(theta)
    theta["k"] = [len(feat[1]) for feat in theta.index]

    print("  Saving additive gauge-fixed coefficients")
    theta_add_model = theta.loc[theta["k"] == 1, :].copy()
    theta_add_model["pos"] = [feat[0][0] for feat in theta_add_model.index]
    theta_add_model["alleles"] = [feat[1] for feat in theta_add_model.index]
    theta_add_model.to_csv(f"results/{dataset_label}.ler.theta_add.csv")

    print("Extracting local pairwise model")
    theta_pw_model = theta.loc[theta["k"] <= 2, :]
    theta_pw_model = pw_model.fixer(theta_pw_model, gauge="zero-sum")
    theta_pw_model["k"] = [len(feat[1]) for feat in theta_pw_model.index]
    print(theta_pw_model)

    print("  Saving constant gauge-fixed coefficients")
    theta0 = (
        theta_pw_model.loc[theta_pw_model["k"] == 0, :]
        .copy()
        .drop("k", axis=1)
        .T
    )
    theta0.to_csv(f"results/{dataset_label}.ler.gauge_fixed_theta_const.csv")

    print("  Saving additive gauge-fixed coefficients")
    theta_add = theta_pw_model.loc[theta_pw_model["k"] == 1, :].copy()
    theta_add["pos"] = [feat[0][0] for feat in theta_add.index]
    theta_add["alleles"] = [feat[1] for feat in theta_add.index]
    theta_add.to_csv(f"results/{dataset_label}.ler.gauge_fixed_theta_add.csv")

    print("  Saving pairwise gauge-fixed coefficients")
    theta_pw = theta_pw_model.loc[theta_pw_model["k"] == 2, :].copy()
    theta_pw["pos1"] = [feat[0][0] for feat in theta_pw.index]
    theta_pw["pos2"] = [feat[0][1] for feat in theta_pw.index]
    theta_pw["allele1"] = [feat[1][0] for feat in theta_pw.index]
    theta_pw["allele2"] = [feat[1][1] for feat in theta_pw.index]
    theta_pw.to_csv(f"results/{dataset_label}.ler.gauge_fixed_theta_pw.csv")

    # print("  Saving all gauge-fixed coefficients")
    # theta.to_csv(f"results/{dataset_label}.ler.gauge_fixed_theta.csv")
    print("Done")
