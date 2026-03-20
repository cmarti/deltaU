import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

if __name__ == "__main__":
    dataset_label = "intron"
    wt = "AGGTACAT"
    positions = np.arange(8)
    position_labels = [2, 3, 4, 5, 18, 19, 20, 21]

    print(f"Plotting contact maps for {dataset_label} dataset")

    print("  Loading gauge-fixed parameters...")
    theta_pw = pd.read_csv(
        f"results/{dataset_label}.ler.gauge_fixed_theta_pw.csv", index_col=0
    )
    
    gauges = [['GNNNNCNC', 'CNNNNCNG'],
              ['GNNNNNCC', 'CNNNNNCG'],
              ['ANNNNCNT', 'TNNNNCNA'],
              ['ANNNNNCT', 'TNNNNNCA']]
    fig, subplots = plt.subplots(
        4, 2, figsize=(5, 2.5 * 4), sharex=True, sharey=True,
    )

    for axes_row, gauge_pair in zip(subplots, gauges):
        for axes, gauge in zip(axes_row, gauge_pair):
            theta_pw[gauge] = np.square(theta_pw[gauge])
            contact_map = theta_pw.groupby(['pos1', 'pos2'])[gauge].mean().reset_index()
            contact_map = pd.pivot(contact_map, index='pos1', columns='pos2', values=gauge)
            contact_map = contact_map.reindex(positions).T.reindex(positions).fillna(0)
            contact_map = contact_map + contact_map.T
            im = axes.imshow(contact_map, cmap="binary", vmin=0, vmax=1)
            axes.set(
                xticks=positions,
                yticks=positions,
                aspect="equal",
                title=gauge,
            )
    for axes in subplots[-1]:
        axes.set(
            xticklabels=position_labels,
            xlabel="Position 1",
        )
    for axes in subplots[:, 0]:
        axes.set(
            yticklabels=position_labels,
            ylabel="Position 2",
        )

    print("  Saving figure...")
    fig.tight_layout()
    fig.savefig(f"figures/{dataset_label}.contact_maps2.png", dpi=300)
    print("Done")
