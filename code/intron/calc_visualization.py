import pandas as pd
from gpmap.randwalk import WMWalk
from gpmap.space import SequenceSpace
from scipy.stats import percentileofscore

if __name__ == "__main__":
    dataset_label = 'intron.30C'
    wt = 'AGGTACAT'
    mean_functions = [0, 0.4, 0.8, 1.2, 1.6, 1.8]
    print(f"Calculating visualization for {dataset_label} dataset")
    
    print("  Loading inferred landscape...")
    data = pd.read_csv(f"results/{dataset_label}.ler.landscape.csv", index_col=0)
    X, f = data.index.values, data.f.values
    wt_fitness = data.loc[wt, 'f']
    wt_perc = percentileofscore(f, wt_fitness)
    print(f'  Fitness of wild-type sequence: {wt_fitness:.2f} ({wt_perc:.2f}% percentile)')

    print("  Calculating visualization at different mean functions")
    space = SequenceSpace(X, f)
    rw = WMWalk(space)
    space.write_edges(f"results/{dataset_label}.edges.npz")
    
    for mean_function in mean_functions:
        perc = percentileofscore(f, mean_function)
        print(f"    Stationary mean function of {mean_function} ({perc:.2f}% percentile)")
        rw.calc_visualization(mean_function=mean_function, n_components=20)
        rw.write_tables(
            prefix=f"results/{dataset_label}.ler.map.mf_{mean_function}",
            nodes_format="pq",
            write_edges=False,
        )
        write_edges = False
    print("Done.")