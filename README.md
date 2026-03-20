# Inference of fitness landscapes with heterogenous patterns of epistasis across sites

This repository contains the code to reproduce the figures and analyses for in our manuscript on Local Epistasis Regression (LER), which is implemented in [gpmap-tools](https://gpmap-tools.readthedocs.io/en/stable/).

- C. Marti-Gomez, D.M. McCandlish. *Inference of fitness landscapes with heterogenous patterns of epistasis across sites* (2025). In preparation.

## Requirements

The analyses were performed using Python 3.8 with the following libraries:

- `gpmap-tools`
- `logomaker`
- `seaborn`

Additionally, this repository requires the following non-Python tools:

- [Snakemake](https://snakemake.readthedocs.io/en/stable/): A workflow management program for improved reproducibility.

## Installation

Create a Conda environment with the required dependencies as follows:

```bash
conda create -n ler python=3.8 
```

Activate the newly created environment and install additional Python dependencies provided in the `requirements.txt` file with the specific versions used in the original analysis:

```bash
conda activate ler
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

## Output

The provided scripts compute all necessary steps from the raw data in the `data` folder and populate the remaining folders in the repository:

- `data/processed`: Contains raw data files downloaded from other sources.
- `data/processed`: Contains intermediate files resulting from processing the raw data.
- `results`: Provides the results of the analyses, typically as `.csv` files.
- `figures`: Stores all panels for the different figures.
