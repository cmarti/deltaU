import gpmap.plot.mpl as mplot
import pandas as pd
import matplotlib



if __name__ == "__main__":
    method = 'ler'
    mf =1
    matplotlib.use("Agg")
    positions_labels = [2, 3, 4, 5, 18, 19, 20, 21]
    nodes_df = pd.read_parquet(f"results/intron.{method}.map.mf_{mf}.nodes.pq")
    mplot.figure_allele_grid(nodes_df, x='1', y='3',
                        nodes_size=5,
                        fpath='figures/intron.alleles.13')