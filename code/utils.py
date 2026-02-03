import numpy as np
import pandas as pd

from itertools import product, combinations


def load_experimental_data(dataset):
    print("Loading training and test experimental data")
    train = pd.read_csv(f"processed/{dataset}.train.csv", index_col=0)
    test = pd.read_csv(f"processed/{dataset}.test.csv", index_col=0)

    X_train, y_train, y_var_train = (
        train.index.values,
        train.y.values,
        train.y_var.values,
    )
    X_test, y_test, y_var_test = (
        test.index.values,
        test.y.values,
        test.y_var.values,
    )
    return (X_train, y_train, y_var_train, X_test, y_test, y_var_test)


def load_sequence_data(species):
    print("\tLoading sequence data from {}".format(species))
    fpath = "processed/{}.seqs.txt".format(species)
    X = np.array([line.strip() for line in open(fpath)])
    return X


def get_contrast_matrix(seqdeft=False, obs=None):
    print("Defining contrasts matrix")
    bc1 = np.array(["".join(x) for x in product("ACGU", repeat=3)])
    bc2 = np.array(["".join(x) for x in product("ACGU", repeat=6)])
    bc3 = np.array(["".join(x) for x in product("ACGU", repeat=9)])
    bc4 = np.array(["".join(x) for x in product("ACGU", repeat=2)])
    p1, p2, p3 = 1.0 / bc1.shape[0], 1.0 / bc2.shape[0], 1.0 / bc3.shape[0]
    p4 = 1.0 / bc4.shape[0]
    contrasts = {
        "AGGAGGNNN": {"AGGAGG{}".format(x): p1 for x in bc1},
        "NGGAGGAGN": {"{}GGAGGAG{}".format(x[0], x[-1]): p4 for x in bc4},
        "NNNAGGNNN": {"{}AGG{}".format(x[:3], x[3:]): p2 for x in bc2},
        "NNNAGGAGG": {"{}AGGAGG".format(x): p1 for x in bc1},
        "AAGGAGGUG": {"AAGGAGGUG": 1.0},
        "NNNNNNNNN": {x: p3 for x in bc3},
    }
    contrasts_matrix1 = pd.DataFrame(contrasts).fillna(0)

    if obs is not None:
        contrasts_matrix1 = contrasts_matrix1.join(obs)

    if seqdeft:
        for col in contrasts_matrix1.columns:
            if col == "NNNNNNNNN":
                continue
            contrasts_matrix1[col] -= contrasts_matrix1["NNNNNNNNN"].values
        contrasts_matrix1.drop("NNNNNNNNN", axis=1, inplace=True)
        # contrasts_matrix1.drop("AAGGAGGUG", axis=1, inplace=True)

    backgrounds = ["UUAAGGAGC", "UAAGGAGCA"]  # , "AAGGAGCAG"
    positions = np.arange(-13, -4)
    contrasts = {}
    for bc1, bc2 in combinations(backgrounds, 2):
        for p, (pos, a1, a2) in enumerate(zip(positions, bc1, bc2)):
            if a1 == a2:
                continue
            label = "{}{}{}".format(a1, pos, a2)

            for bc in [bc1, bc2]:
                s = [c for c in bc]
                s[p] = a1
                s1 = "".join(s)
                s[p] = a2
                s2 = "".join(s)
                contrasts["{}_in_{}".format(label, bc)] = {s1: -1, s2: 1}
    contrasts_matrix2 = pd.DataFrame(contrasts).fillna(0)
    contrasts_matrix = pd.concat([contrasts_matrix1, contrasts_matrix2]).fillna(
        0
    )
    if seqdeft:
        contrasts_matrix = -contrasts_matrix

    return contrasts_matrix
