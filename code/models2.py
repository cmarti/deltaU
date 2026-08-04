
from itertools import product, combinations

import numpy as np
import pandas as pd
from gpmap.linop import StackedOperator, KronOperator, SelIdxOperator, DiagonalOperator
from scipy.stats import pearsonr
from scipy.linalg import orth
from scipy.sparse.linalg import minres


class TruncatedBasisOperator(StackedOperator):
    def __init__(self, alphas, max_k):
        self.max_k = max_k
        self.alphas = alphas
        self.seq_length = len(alphas)
        self.positions = np.arange(self.seq_length)
        
        self.site_Ls = [alpha * np.eye(alpha) - np.ones((alpha, alpha)) for alpha in alphas]
        self.site_basis = [[np.full((alpha, 1), 1 / np.sqrt(alpha)), orth(site_L)]
                      for alpha, site_L in zip(self.alphas, self.site_Ls)]
        
        As = []
        rank = 0
        for k in range(self.max_k + 1):
            for j in combinations(self.positions, k):
                matrices = [b[int(i in j)] for i, b in enumerate(self.site_basis)]
                A = KronOperator(matrices)
                As.append(A)
                rank += A.shape[1]
        self.rank = rank
        super().__init__(linops=As, axis=1)
    
    def transpose(self):
        As = []
        for k in range(self.max_k + 1):
            for j in combinations(self.positions, k):
                matrices = [b[int(i in j)].T for i, b in enumerate(self.site_basis)]
                A = KronOperator(matrices)
                As.append(A)
        return StackedOperator(linops=As, axis=0)


class TruncatedModel:
    def __init__(self, genotypes, max_k):
        alleles = np.array([[c for c in s] for s in genotypes])
        self.max_k = max_k
        self.alphabets = [np.unique(x) for x in alleles.T]
        self.alphas = [alphabet.shape[0] for alphabet in self.alphabets]
        self.genotypes = np.array([''.join(g) for g in product(*self.alphabets)])
        self.basis = TruncatedBasisOperator(alphas=self.alphas, max_k=self.max_k)
        self.genotypes_idx = {g: i for i, g in enumerate(self.genotypes)}
        self.n_params = self.basis.shape[1]
        self.beta = np.zeros(self.n_params)
    
    def fit(self, X, y, y_var=None):
        idx = np.array([self.genotypes_idx[g] for g in X])
        n = self.basis.shape[0]
        S = SelIdxOperator(n, idx)
        X = S @ self.basis
        Xt = self.basis.transpose() @ S.transpose()
        
        if y_var is None or np.allclose(y_var, 0):
            A = Xt @ X
            b = Xt @ y
        else:
            D_var_inv = DiagonalOperator(1 /y_var)
            A = Xt @ D_var_inv @ X
            b = Xt @ D_var_inv @ y
        self.result = minres(A, b)
        self.beta = self.result[0]
    
    def predict(self):
        return pd.DataFrame({'f': self.basis @ self.beta}, index=self.genotypes)


def evaluate_predictions(y_pred, X_train, X_test, y_train, y_test, label, p):
    try:
        f_train_pred = y_pred.loc[X_train, "f"].values
        r2_train = pearsonr(f_train_pred, y_train)[0] ** 2  # type: ignore
        rmse_train = np.sqrt(np.mean((y_train - f_train_pred) ** 2))
        mae_train = np.mean(np.abs(y_train - f_train_pred))
    except KeyError:
        r2_train = np.nan
        rmse_train = np.nan
        mae_train = np.nan
    
    f_test_pred = y_pred.loc[X_test, "f"].values
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
