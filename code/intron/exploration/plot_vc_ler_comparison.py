from code.plot_utils import (
    apply_plot_style,
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __name__ == "__main__":
    dataset_label = 'intron.30C'
    apply_plot_style()
    
    print('  Loading data...')
    ler = pd.read_csv(f'results/{dataset_label}.ler.pred.csv', index_col=0)
    vc = pd.read_csv(f'results/{dataset_label}.vc.pred.csv', index_col=0)
    print(np.std(np.log(ler['f_var'])), np.std(np.log(vc['f_var'])))
    
    
    fig, subplots = plt.subplots(1, 2, figsize=(5, 2))
    subplots = subplots.flatten()
    
    axes = subplots[0]
    axes.scatter(ler['f'], vc['f'], alpha=0.5, s=5, c='black', lw=0)
    axes.set(xlabel='LER predictions', ylabel='VC predictions', aspect='equal')
    axes.axline((0, 0), (1, 1), color='gray', ls='--', lw=0.75)
    
    axes = subplots[1]
    axes.scatter(ler['f_var'], vc['f_var'], alpha=0.5, s=5, c='black', lw=0)
    axes.set(xlabel='LER prediction variances', ylabel='VC prediction variances', aspect='equal')
    axes.axline((0.7, 0.7), (1, 1), color='gray', ls='--', lw=0.75)
        
    plt.tight_layout()
    plt.savefig(f"figures/{dataset_label}.models_posterior_comparison.png", dpi=300)
    
    print('Done.')