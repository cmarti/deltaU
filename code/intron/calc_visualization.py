import pandas as pd

from gpmap.randwalk import WMWalk
from gpmap.space import SequenceSpace


if __name__ == "__main__":
    dataset_label = 'intron.30C'
    print(f"Calculating visualization for {dataset_label} dataset")
    
    print("  Loading inferred landscape...")
    data = pd.read_csv(f"results/{dataset_label}.ler.landscape.csv", index_col=0)
    X, f = data.index.values, data.f.values

    print("  Calculating visualization at different mean functions")
    space = SequenceSpace(X, f)
    rw = WMWalk(space)
    space.write_edges(f"results/{dataset_label}.edges.npz")
    
    for mean_function in [1.9]:# [0, 0.4, 0.8, 1.2, 1.6, 2]:
        print("    Stationary mean function of {}".format(mean_function))
        rw.calc_visualization(mean_function=mean_function, n_components=20)
        rw.write_tables(
            prefix=f"results/{dataset_label}.ler.map.mf_{mean_function}",
            nodes_format="pq",
            write_edges=False,
        )
        write_edges = False
    print("Done.")