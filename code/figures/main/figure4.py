from code.plot_utils import (
    FIG_WIDTH,
    POSITION_LABELS,
    add_panel_labels,
    apply_plot_style,
    plot_correlation_landscape,
    plot_interaction_matrix,
    plot_pred_vs_obs_corr,
    plot_site_pairs_variance_components,
    plot_sites_variance_components,
    plot_test_pred_comparison,
    plot_train_pred_comparison,
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __name__ == "__main__":
    dataset_name = "intron.30C"
    position_labels = POSITION_LABELS[dataset_name]
    apply_plot_style()

    print(f"Plotting model fit for {dataset_name} dataset")

    print("Loading data for plotting")
    print("  Loading correlation data...")
    nodes_df = pd.read_csv(
        f"results/{dataset_name}.corrs.csv",
        dtype={"seq": str},
        index_col="seq",
    )

    print("  Loading a matrix...")
    a_matrix = pd.read_csv(
        f"results/{dataset_name}.interaction_strength.csv", index_col=0
    )
    
    print('  Loading predictions in training set...')
    landscape = pd.read_csv(f'results/{dataset_name}.ler.landscape.csv', index_col=0)
    train = pd.read_csv(f'data/processed/{dataset_name}.train.csv', index_col=0)
    train = train.join(landscape)
    
    print('  Loading predictions in test set...')
    pred = pd.read_csv(f'results/{dataset_name}.ler.pred.csv', index_col=0)
    data = pd.read_csv(f'data/processed/{dataset_name}.test.csv', index_col=0)
    pred = pred.join(data)
    pred['y_std'] = np.sqrt(pred['y_var'])
    coverage = np.mean((pred['y'] > pred['ci_95_lower']) & (pred['y'] < pred['ci_95_upper']))
    print(f'    Total test sequences: {pred.shape[0]}')
    print(f'    Coverage of 95% CI: {coverage*100:.2f}')
    
    print('  Loading R2 curves data')
    r2 = pd.read_csv(f"results/{dataset_name}.r2.csv", index_col=0)
    print(r2)
    
    print("  Loading variance explained by interactions of order k for site i")
    fpath = f'results/{dataset_name}.ler.sites_variance_k.csv'
    sites = pd.read_csv(fpath, index_col=0)
    
    print("  Loading variance explained by interactions of order k=2 and k>2 for pairs of sites")
    fpath = f'results/{dataset_name}.ler.sites_pairs_variance.csv'
    m = pd.read_csv(fpath, index_col=0)

    print("Making figure...")
    fig, subplots = plt.subplots(2, 4, figsize=(1.12*FIG_WIDTH, FIG_WIDTH * 0.45))

    axes = subplots[0, 0]
    axes.axis('off')

    print("  Plotting correlation landscape...")
    axes = subplots[0, 1]
    plot_correlation_landscape(nodes_df, axes, y="emp_cor")
    axes.set(ylabel="Observed correlation")

    print("  Plotting predicted vs observed correlations...")
    axes = subplots[0, 2]
    plot_pred_vs_obs_corr(nodes_df, axes)

    print("  Plotting a matrix...")
    axes = subplots[0, 3]
    plot_interaction_matrix(
        a_matrix,
        axes,
        vmax=None,
        position_labels=POSITION_LABELS[dataset_name],
    )
    
    print("  Plotting predictions in training sequences")
    axes = subplots[1, 0]
    plot_train_pred_comparison(train, axes, lims=(-8, 6))
    
    print("  Plotting predictions in held-out sequences")
    axes = subplots[1, 1]
    plot_test_pred_comparison(pred, axes, lims=(-8, 6))
    
    # print("  Plotting cross-validation curves")
    # axes = subplots[1, 1]
    # plot_cv_r2_curves(r2, axes)
    # axes.set(ylim=(0.2, 0.8))
    
    print("  Plotting variance explained by interactions of order k for site i")
    axes = subplots[1, 2]
    plot_sites_variance_components(axes, sites)
    
    print("  Plotting variance explained by interactions of order k=2 and k>2 for pairs of sites")
    axes = subplots[1, 3]
    print(m)
    plot_site_pairs_variance_components(axes, m)

    print("  Saving figure...")
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.8)
    add_panel_labels(subplots.flatten(), labels=['A', 'B', 'C', 'D','E', 'F', 'G', 'H'], x_offset=-0.32, y_offset=1.075)
    fig.savefig("figures/figure4.png", dpi=300)
    fig.savefig("figures/figure4.svg", dpi=300)

    print("Done.")
