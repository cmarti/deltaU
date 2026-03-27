
import numpy as np
import pandas as pd
from pydeseq2.dds import DeseqDataSet
from pydeseq2.default_inference import DefaultInference
from pydeseq2.ds import DeseqStats

if __name__ == "__main__":
    np.random.seed(0)

    print("Loading counts data")
    fpath = "data/raw/intron.csv"
    data = pd.read_csv(fpath, index_col=0)
    counts = data.iloc[:, -24:]

    print("Preparing metadata")
    samples = counts.columns
    samples_group = np.array([c.split("_")[0] for c in samples])
    groups = np.unique(samples_group)
    metadata = pd.DataFrame({"group": samples_group}, index=samples)

    print("Running PyDESeq2")
    inference = DefaultInference(n_cpus=8)
    dds = DeseqDataSet(
        counts=counts.T,
        metadata=metadata,
        design_factors="group",
        refit_cooks=True,
        inference=inference,
    )
    dds.fit_size_factors()
    dds.fit_genewise_dispersions()
    dds.fit_dispersion_trend()
    dds.fit_dispersion_prior()
    dds.fit_MAP_dispersions()
    dds.fit_LFC()
    dds.calculate_cooks()
    dds.refit()

    print("Computing contrasts")
    for temp in [30, 37]:
        print(f"  For temperature: {temp}C")
        ds = DeseqStats(
            dds,
            contrast=["group", f"Kan+{temp}C", f"Kan-{temp}C"],
            alpha=0.05,
            cooks_filter=True,
            independent_filter=True,
        )
        ds.run_wald_test()
        ds.summary()
        df = pd.DataFrame(
            {
                "y": ds.results_df["log2FoldChange"],
                "y_var": ds.results_df["lfcSE"] ** 2,
            },
            index=ds.results_df.index.values,
        )
        df.to_csv(f"data/processed/intron.{temp}C.csv")
    print("Done.")
