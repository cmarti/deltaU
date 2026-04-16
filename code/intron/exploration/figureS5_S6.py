from code.plot_utils import (
    FIG_WIDTH,
    add_r2_label,
    apply_plot_style,
    plot_train_pred_comparison,
)
from itertools import combinations

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr

if __name__ == "__main__":
    dataset_label = 'intron.30C'
    apply_plot_style()
    
    print(f"Comparing all models on {dataset_label} data")
    print('  Loading data...')
    data = pd.read_csv(f'results/{dataset_label}.models_predictions.csv', index_col=0)
    
    
    print("  Plotting model comparisons in full landscape")
    fig, subplots = plt.subplots(3, 5, figsize=(1.4 * FIG_WIDTH, 0.65 * FIG_WIDTH))
    subplots = subplots.flatten()
    
    for axes, (model1, model2) in zip(subplots, combinations(data.columns, 2)):
        plot_train_pred_comparison(data, axes, lims=(-8, 6), x=model1, y=model2)
        axes.set(xlabel=f'{model1} predictions', ylabel=f'{model2} predictions', 
                 aspect='equal')
        r2 = pearsonr(data[model1], data[model2])[0] ** 2
        add_r2_label(axes, r2)
        
    plt.tight_layout()
    plt.savefig("figures/figureS5.png", dpi=300)
    
    print("  Plotting model comparisons in held-out data")
    test = pd.read_csv(f"data/processed/{dataset_label}.test.csv", index_col=0)
    data = data.loc[test.index, :]
    
    fig, subplots = plt.subplots(3, 5, figsize=(FIG_WIDTH, 0.6 * FIG_WIDTH),
                                 sharex=True, sharey=True)
    subplots = subplots.flatten()
    
    for axes, (model1, model2) in zip(subplots, combinations(data.columns, 2)):
        axes.scatter(data[model1], data[model2], alpha=0.5, s=3, c='black', lw=0)
        axes.axline((0, 0), (1, 1), color='gray', ls='--', lw=0.75)
        axes.set(xlabel=f'{model1} predictions', ylabel=f'{model2} predictions', 
                 aspect='equal', xlim=(-4.5, 3.5), ylim=(-4.5, 3.5))
        r2 = pearsonr(data[model1], data[model2])[0] ** 2
        add_r2_label(axes, r2)
        
    plt.tight_layout()
    plt.savefig("figures/figureS6.png", dpi=300)
    print('Done.')