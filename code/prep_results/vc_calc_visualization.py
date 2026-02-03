import pandas as pd

from gpmap.randwalk import WMWalk
from gpmap.space import SequenceSpace


if __name__ == "__main__":
    method = 'ler'
    
    print("Loading inferred landscape")
    data = pd.read_csv(f"results/intron.{method}.landscape.csv", index_col=0)
    print(data)
    X, y = data.index.values, data.f.values
    print(y.mean(), y.max())

    print("Calculating visualization for VC regression MAP")
    space = SequenceSpace(X, y)
    rw = WMWalk(space)
    write_edges = True
    for mean_function in [-1.5]:
        print("\tStationary mean function of {}".format(mean_function))
        rw.calc_visualization(mean_function=mean_function, n_components=20)
        rw.write_tables(
            prefix=f"results/intron.{method}.map.mf_{mean_function}",
            nodes_format="pq",
            write_edges=write_edges,
        )
        write_edges = False
