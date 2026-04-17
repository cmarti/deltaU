# Inference of fitness landscapes with heterogenous patterns of epistasis across sites

This repository contains the code to reproduce the figures and analyses for in our manuscript on Local Epistasis Regression (LER), which is implemented in [gpmap-tools](https://gpmap-tools.readthedocs.io/en/stable/).

- C. Marti-Gomez, D.M. McCandlish. *Inference of fitness landscapes with heterogenous patterns of epistasis across sites* (2025). In preparation.

## Requirements

The analyses were performed using Python 3.8 with the following libraries:

- `gpmap-tools`
- `pydeseq2`

Additionally, this repository requires the following non-Python tools:

- [Snakemake](https://snakemake.readthedocs.io/en/stable/): A workflow management program for improved reproducibility.

## Installation

Create a Conda environment with the required dependencies as follows:

```bash
conda create -n deltaU python=3.8 
```

Activate the newly created environment and install additional Python dependencies provided in the `requirements.txt` file with the specific versions used in the original analysis:

```bash
conda activate deltaU
pip install -r requirements.txt
```

This setup should be sufficient to run the provided Python scripts in the `code` folder.

For users who wish to reproduce all or subsets of the figures automatically, we provide an additional Snakemake workflow in the `Snakefile`. To use it, create a second environment for installing Snakemake:

```bash
conda create -n snakemake bioconda::snakemake
conda activate snakemake
```

## Reproducing the Results

Once installed, you can run the full workflow with:

```bash
snakemake --snakefile Snakefile --cores 1 --use-conda all
```

Alternatively, you can generate only the main or supplementary figures:

```bash
snakemake --snakefile Snakefile --cores 1 --use-conda main_figures
snakemake --snakefile Snakefile --cores 1 --use-conda supplementary_figures
```

You can also generate panels for specific figures of interest, for example:

```bash
snakemake --snakefile Snakefile --cores 1 --use-conda figure3
```

Note: computation of the R2 values for different amounts of training data is computationally expensive. We provide the R2 values in the repository and the code to obtain it is commented out in the Snakefile.

## Output

The provided scripts compute all necessary steps from the raw data in the `data` folder and populate the remaining folders in the repository:

- `data/raw`: Contains raw data files downloaded from other sources.
- `data/processed`: Contains intermediate files resulting from processing the raw data.
- `results`: Provides the results of the analyses, typically as `.csv` files.
- `figures`: Stores all panels for the different figures.
