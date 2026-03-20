import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

if __name__ == "__main__":
    dataset_label = "intron"
    wt = "AGGTACAT"
    positions = np.arange(8)
    position_labels = [2, 3, 4, 5, 18, 19, 20, 21]

    print(f"Plotting gauge-fixed parameters for {dataset_label} dataset")

    print("  Loading gauge-fixed parameters...")
    theta_add = pd.read_csv(
        f"results/{dataset_label}.ler.gauge_fixed_theta_add.csv", index_col=0
    )
    print(theta_add)

    
    gauges = [['GC', 'CG'],
              ['GT', 'TG'],
              ['AT', 'TA']]
    fig, subplots = plt.subplots(
        3, 2, figsize=(5, 1.2 * 3), sharex=True, 
    )

    for axes_row, gauge_pair in zip(subplots, gauges):
        for axes, gauge in zip(axes_row, gauge_pair):
            theta_m = pd.pivot_table(
                theta_add, index="alleles", columns="pos", values=gauge
            )
            im = axes.imshow(theta_m, cmap="coolwarm", vmin=-2, vmax=2)
            # plt.colorbar(im, label='parameter')
            axes.set(
                ylabel=gauge[0] + r"$_2$" + gauge[1] + r'$_{21}$',
                xticks=positions,
                aspect="equal",
                yticks=np.arange(4),
                yticklabels=theta_m.index.values,
            )
    for axes in subplots[-1]:
        axes.set(
            xticklabels=position_labels,
            xlabel="Position",
        )

    print("  Saving figure...")
    fig.tight_layout()
    fig.savefig(f"figures/{dataset_label}.gauge_fixed_theta.png", dpi=300)

    print("Done")
