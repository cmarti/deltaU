import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import pearsonr


if __name__ == "__main__":
    np.random.seed(0)
    fpath = "data/raw/intron.csv"
    wt = 'AGGTACAT'
    data = pd.read_csv(fpath, index_col=0)
    
    cols = []
    for temp in [30, 37]:
        for rep in range(1, 6):
            col = f'{temp}C_R{rep}'
            cols.append(col)
            idx = data[f'Kan-{temp}C_R{rep}'] > 20
            c = data.loc[idx, :]
            
            c0i = c[f'Kan-{temp}C_R{rep}'] + 0.5
            c0wt = c0i.loc[wt]
            c1i = data[f'Kan+{temp}C_R{rep}'] + 0.5
            c1wt = data[f'Kan+{temp}C_R{rep}'].loc[wt]
            data[col] = np.log2(c1i / c1wt) - np.log2(c0i / c0wt)
            data[f'{col}_var'] = 1 / c0i + 1/c0wt + 1/c1i + 1/c1wt
    n = len(cols)
    data[cols].to_csv('data/processed/intron.replicates.csv')
    
    processed = {}
    for temp in [30, 37]:
        cs = [f'{temp}C_R{i}_var' for i in range(1, 6)]
        ws = (1 / data[cs]).fillna(0)
        y_var_inv = np.nansum(ws.values, axis=1)
        y_var = 1 / y_var_inv
        cs = [f'{temp}C_R{i}' for i in range(1, 6)]
        y = np.nansum(data[cs].values * ws.values, axis=1) / y_var_inv
        processed[f'{temp}C_y'] = y
        processed[f'{temp}C_y_var'] = y_var
    
    processed = pd.DataFrame(processed, index=data.index).dropna()
    
    U, s, V = np.linalg.svd(processed[['30C_y', '37C_y']].values, full_matrices=False)
    processed['37C_y'] = processed['37C_y'] / V[0, 0]
    processed['37C_y_var'] = processed['37C_y_var'] / V[0, 0] ** 2
    processed['37C_y'] += (processed['30C_y'] - processed['37C_y']).mean()
    processed.to_csv('data/processed/intron.csv')
    
    idx = np.random.uniform(size=processed.shape[0]) < 0.005
    train = processed.loc[~idx, :]
    test = processed.loc[idx, :]
    train.to_csv("data/processed/intron.train.csv")
    test.to_csv("data/processed/intron.test.csv")
    exit()
    
    bins = np.linspace(-10, 2, 50)
    fig, axes = plt.subplots(1, 1, figsize=(3, 3))
    sns.histplot(x=processed['30C_y'], y=processed['37C_y'], cmap='binary', bins=[bins, bins], ax=axes)
    r = pearsonr(processed['30C_y'], processed['37C_y'])[0]
    axes.text(0.05, 0.95, r'$\rho$' + f'={r:.2f}', transform=axes.transAxes,
            ha='left', va='top')
    axes.axline((0, 0), slope=1, lw=0.5, linestyle='--', c='grey')
    axes.axvline(0, lw=0.5, linestyle='--', c='grey')
    axes.axhline(0, lw=0.5, linestyle='--', c='grey')
            
    axes.set(ylabel='30C', xlabel='37C')
    fig.tight_layout()
    fig.savefig('figures/conditions.png', dpi=300)
    
    fig, subplots = plt.subplots(n, n, figsize=(12, 12), sharex=True, sharey=True)
    
    bins = np.linspace(-10, 2, 50)
    for v1, ax_row in zip(cols, subplots):
        for v2, axes in zip(cols, ax_row):
            # if v1 == v2:
            #     axes.hist(data[v1], bins=bins, color='grey', alpha=0.5)
            # else:
            if v1 != v2:
                # axes.scatter(data[v1], data[v2], s=5, c='black', alpha=0.1, lw=0)
                df = data[[v1, v2]].dropna()
                sns.histplot(x=df[v1].values, y=df[v2].values, cmap='binary', bins=[bins, bins], ax=axes)
                r = pearsonr(df[v1], df[v2])[0]
                axes.text(0.05, 0.95, r'$\rho$' + f'={r:.2f}', transform=axes.transAxes,
                        ha='left', va='top')
            axes.axline((0, 0), slope=1, lw=0.5, linestyle='--', c='grey')
            axes.axvline(0, lw=0.5, linestyle='--', c='grey')
            axes.axhline(0, lw=0.5, linestyle='--', c='grey')
            
    for axes, col in zip(subplots[:, 0], cols):
        axes.set(ylabel=col)
    
    for axes, col in zip(subplots[-1, :], cols):
        axes.set(xlabel=col)
        
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    fig.savefig('figures/replicates.png', dpi=300)
            
        
    
    
    