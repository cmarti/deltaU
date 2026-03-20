import gpmap.plot.ply as pplot
import pandas as pd


if __name__ == "__main__":
    dataset_label = "intron.30C"
    mf = 1
    x, y, z = 1, 2, 3
    print(f"Plotting interactive visualization for {dataset_label} dataset")

    print("  Loading input data")
    nodes_df = pd.read_parquet(
        f"results/{dataset_label}.ler.map.mf_{mf}.nodes.pq"
    )
    print("  Selecting top 10,000 genotypes")
    nodes_df = nodes_df.sort_values("function", ascending=False).iloc[:5000, :]

    print("  Plotting visualization")
    pplot.plot_visualization(
        nodes_df,
        x=str(x),
        y=str(y),
        z=str(z),
        nodes_cmap_label="Fitness",
        fpath=f"figures/{dataset_label}.ler.visualization",
    )

    print("Done.")
