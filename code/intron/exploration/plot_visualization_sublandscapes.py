import gpmap.plot.mpl as mplot
import pandas as pd
import matplotlib.pyplot as plt

from gpmap.randwalk import WMWalk
from gpmap.space import SequenceSpace

if __name__ == "__main__":
    dataset_label = "intron.30C"
    mf = 1
    x, y, z = 1, 2, 3
    print(f"Plotting interactive visualization for {dataset_label} dataset")

    print("  Loading input data")
    data = pd.read_csv(
        f"results/{dataset_label}.ler.landscape.csv", index_col=0
    )
    data["21"] = [x[-1] for x in data.index.values]
    data["seq"] = [x[:-1] for x in data.index.values]

    fig, subplots = plt.subplots(2, 4, figsize=(12, 4.5))
    for ax_col, (allele, df) in zip(subplots.T, data.groupby("21")):
        df.set_index("seq", inplace=True)
        X, f = df.index.values, df.f.values

        space = SequenceSpace(X, f)
        rw = WMWalk(space)
        rw.calc_visualization(mean_function=2.2)
        nodes_df = rw.nodes_df

        print("Allele", allele)
        if allele in ["C", "T"]:
            print(nodes_df.sort_values("1").tail(10))
        else:
            print(nodes_df.sort_values("2").tail(10))

        print("  Plotting visualization")
        vmin, vmax = -5, 4
        mplot.plot_nodes(
            ax_col[0],
            nodes_df,
            x=str(x),
            y=str(y),
            sort_by=str(z),
            sort_ascending=False,
            size=1.5,
            vmin=vmin,
            vmax=vmax,
            cmap="viridis",
            cbar_label="Fitness",
            rasterized=True,
        )

        mplot.plot_nodes(
            ax_col[1],
            nodes_df,
            x=str(x),
            y=str(z),
            sort_by=str(y),
            sort_ascending=False,
            size=1.5,
            vmin=vmin,
            vmax=vmax,
            cmap="viridis",
            cbar_label="Fitness",
            rasterized=True,
        )

    fig.tight_layout()
    fig.savefig(
        f"figures/{dataset_label}.visualization.sublandscapes.png", dpi=300
    )
    print("Done.")
