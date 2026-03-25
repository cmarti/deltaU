from itertools import product

import numpy as np
import pandas as pd
from gpmap.inference import LocalEpistasisRegression
from gpmap.matrix import kron


def get_contrast_matrix():
    alphabet = list('ACGU')
    alleles = np.eye(4)
    alleles_dict = dict(zip(alphabet, alleles))
    background = np.full((4, 1), 1/4.)
    seqs = np.array([''.join(x) for x in product(alphabet, repeat=8)])
    
    # First set of contrasts
    columns = [f'2{a1}_21{a2}' for a1, a2 in product(alphabet, repeat=2)]
    c1 = kron([alleles] + [background] * 6 + [alleles])
    c1 = pd.DataFrame(c1, index=seqs, columns=columns)
    cs = [c1]
    
    # Second set of contrasts
    backgrounds = ['GC', 'CG', 'AU', 'UA']
    alleles_target = alleles[:, 1:3]
    alleles_target_labels = 'CG'
    for bc in backgrounds:
        b2 = alleles_dict[bc[0]].reshape((4, 1)) 
        b21 = alleles_dict[bc[1]].reshape((4, 1))
        factors = [b2, alleles_target] + [background] * 4 + [alleles_target, b21]
        c = kron(factors)
        columns = [f'2{bc[0]}_3{a3}_20{a20}_21{bc[1]}'
                   for a3, a20 in product(alleles_target_labels, repeat=2)]
        c = pd.DataFrame(c, index=seqs, columns=columns)
        cs.append(c)
        
    # Third set of contrasts
    backgrounds = ['GC', 'CG', 'AU', 'UA', 'AC', 'AG']
    alleles_target = alleles[:, 1:3]
    alleles_target_labels = 'CG'
    for bc in backgrounds:
        b2 = alleles_dict[bc[0]].reshape((4, 1)) 
        b21 = alleles_dict[bc[1]].reshape((4, 1))
        factors = [b2] + [background] * 4 + [alleles_target, alleles_target, b21]
        c = kron(factors)
        columns = [f'2{bc[0]}_19{a19}_20{a20}_21{bc[1]}'
                   for a19, a20 in product(alleles_target_labels, repeat=2)]
        c = pd.DataFrame(c, index=seqs, columns=columns)
        cs.append(c)
    
    c = pd.concat(cs, axis=1)
    return(c)
    


if __name__ == "__main__":
    dataset_label = 'intron.30C'
    print(f"Making contrasts with Local Epistasis Regression model fitted to {dataset_label} data")
    
    print('  Loading data...')
    data = pd.read_csv(f"data/processed/{dataset_label}.csv", index_col=0)
    data.index = [x.replace('T', 'U') for x in data.index]
    
    contrast_matrix = get_contrast_matrix()
    X, y, y_var = data.index.values, data["y"].values, data["y_var"].values

    print('  Loading model parameters...')
    fpath = f'results/{dataset_label}.ler.a.npy'
    a_values = np.load(fpath)
    print(f"    Loaded a_values: {a_values}")

    fpath = f'results/{dataset_label}.ler.lambda_U.npy'
    lambda_U = np.load(fpath)
    print(f"    Loaded lambda_U: {lambda_U}")

    print('  Making contrasts...')
    model = LocalEpistasisRegression(seq_length=8, alphabet_type="dna", P=2,
                                     a_values=a_values, lambda_U_lower_than_P=lambda_U)
    model.set_data(X, y, y_var=y_var)
    contrasts = model.make_contrasts(contrast_matrix)
    
    print('  Saving contrasts...')
    contrasts.to_csv(f'results/{dataset_label}.ler.contrasts.csv')
    print('Done.')