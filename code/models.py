
from itertools import product

import numpy as np
import pandas as pd
from gpmap.linop import DeltaKernelBasisOperator
from scipy.stats import pearsonr


class TruncatedModel:
    def __init__(self, genotypes, max_k):
        self.alphabet = np.unique([[c for c in s] for s in genotypes])
        self.seq_length = len(genotypes[0])
        self.n_alleles = len(self.alphabet)
        self.genotypes = np.array([''.join(g) for g in product(self.alphabet, repeat=self.seq_length)])
        
        self.max_k = max_k
        self.basis = DeltaKernelBasisOperator(
            n_alleles=self.n_alleles, seq_length=self.seq_length, P=self.max_k + 1
        ).todense()
        self.genotypes_idx = {g: i for i, g in enumerate(self.genotypes)}
        self.n_params = self.basis.shape[1]
        self.beta = np.zeros(self.n_params)
    
    def fit(self, X, y, y_var=None):
        idx = [self.genotypes_idx[g] for g in X]
        X = self.basis[idx, :]
        self.V = None
        if np.linalg.matrix_rank(X) < X.shape[1]:
            print('Taking SVD decomposition for basis')
            X, _, self.V = np.linalg.svd(X, full_matrices=False)
        
        if y_var is None or np.allclose(y_var, 0):
            A = X.T @ X
            b = X.T @ y
        else:
            A = X.T @ (X / y_var[:, None])
            b = X.T @ (y / y_var)
        self.beta = np.linalg.solve(A, b)
    
    def predict(self):
        X = self.basis if self.V is None else self.basis @ self.V.T
        f = X @ self.beta
        return pd.DataFrame({'f': f}, index=self.genotypes)


def evaluate_predictions(y_pred, X_train, X_test, y_train, y_test, label, p):
    f_train_pred = y_pred.loc[X_train, "f"].values
    f_test_pred = y_pred.loc[X_test, "f"].values

    r2_train = pearsonr(f_train_pred, y_train)[0] ** 2  # type: ignore
    rmse_train = np.sqrt(np.mean((y_train - f_train_pred) ** 2))
    mae_train = np.mean(np.abs(y_train - f_train_pred))
    
    r2_test = pearsonr(f_test_pred, y_test)[0] ** 2  # type: ignore
    rmse_test = np.sqrt(np.mean((y_test - f_test_pred) ** 2))
    mae_test = np.mean(np.abs(y_test - f_test_pred))
    record = {
        "p": p,
        "r2_train": r2_train,
        "rmse_train": rmse_train,
        "mae_train": mae_train,
        "r2_test": r2_test,
        "rmse_test": rmse_test,
        "mae_test": mae_test,
        "model": label,
    }
    return(record)
