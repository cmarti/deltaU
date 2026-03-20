import gpmap.plot.ply as pplot
import pandas as pd

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

    for allele, df in data.groupby("21"):
        df.set_index("seq", inplace=True)
        X, f = df.index.values, df.f.values

        space = SequenceSpace(X, f)
        rw = WMWalk(space)
        rw.calc_visualization(mean_function=1.6)

        nodes_df = rw.nodes_df.sort_values("function", ascending=False).iloc[
            :5000, :
        ]

        print("  Plotting visualization")
        pplot.plot_visualization(
            nodes_df,
            x=str(x),
            y=str(y),
            z=str(z),
            nodes_cmap_label="Fitness",
            fpath=f"figures/{dataset_label}.visualization.21{allele}",
        )
    print("Done.")
