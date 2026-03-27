shell.prefix("source activate.sh ; source $(conda info --base)/etc/profile.d/conda.sh ; conda activate deltaU ; ")

INTRON_MFS = [0, 0.4, 0.8, 1.2, 1.6, 1.8]
INTRON_MF_SOURCE = 1.9

rule figure2:
    input:
        "figures/figure2.png",

rule figure3:
    input:
        "figures/figure3.png",

rule figure4:
    input:
        "figures/figure4.png",

rule figure5:
    input:
        "figures/figure5a.png",
        "figures/figure5bch.png",
        "figures/figure5defg.png",

rule figureS1:
    input:
        "figures/figureS1.png",

rule figureS2:
    input:
        "figures/figureS2.png",

rule figureS3:
    input:
        "figures/figureS3.png",

rule main_figures:
    input:
        rules.figure2.input,
        rules.figure3.input,
        rules.figure4.input,
        rules.figure5.input,

rule supplementary_figures:
    input:
        rules.figureS1.input,
        rules.figureS2.input,
        rules.figureS3.input,

rule all:
    input:
        rules.main_figures.input,
        rules.supplementary_figures.input


# Simulations (Figure 2)
rule simulate_data:
    output:
        "data/processed/simulations.csv",
        "results/simulations.prior_correlations.csv",
        "results/simulations.prior_a.csv",
    shell:
        "python code/simulations/simulate.py"

rule split_simulated_data:
    input:
        "data/processed/simulations.csv",
    output:
        "data/processed/simulations.train.csv",
        "data/processed/simulations.test.csv",
    shell:
        "python code/simulations/split_train_test.py"

rule fit_simulated_data:
    input:
        "data/processed/simulations.train.csv",
        "data/processed/simulations.test.csv",
    output:
        "results/simulations.corrs.csv",
        "results/simulations.inferred_interaction_strength.csv",
        "results/simulations.pred.csv",
    shell:
        "python code/simulations/fit.py"

rule calc_simulation_r2:
    input:
        "data/processed/simulations.csv",
    output:
        "results/simulations.r2.csv",
    shell:
        "python code/simulations/calc_r2_curves.py"

rule plot_figure2:
    input:
        "results/simulations.prior_a.csv",
        "results/simulations.prior_correlations.csv",
        "results/simulations.corrs.csv",
        "results/simulations.r2.csv",
    output:
        "figures/figure2.png",
        "figures/figure2.svg",
    shell:
        "python code/figures/main/figure2.py"


# Public datasets (Figure 3, Figure S1)
rule process_fyn_sh3:
    input:
        "data/raw/fyn-sh3.csv",
    output:
        "data/processed/fyn-sh3.csv",
    shell:
        "python code/datasets/process_fyn-sh3.py"

rule split_datasets:
    input:
        "data/processed/fyn-sh3.csv",
    output:
        "data/processed/smn1.train.csv",
        "data/processed/smn1.test.csv",
        "data/processed/dmsc.train.csv",
        "data/processed/dmsc.test.csv",
        "data/processed/gb1.train.csv",
        "data/processed/gb1.test.csv",
        "data/processed/fyn-sh3.train.csv",
        "data/processed/fyn-sh3.test.csv",
    shell:
        "python code/datasets/split_train_test.py"

rule fit_datasets:
    input:
        "data/processed/smn1.train.csv",
        "data/processed/smn1.test.csv",
        "data/processed/dmsc.train.csv",
        "data/processed/dmsc.test.csv",
        "data/processed/gb1.train.csv",
        "data/processed/gb1.test.csv",
        "data/processed/fyn-sh3.train.csv",
        "data/processed/fyn-sh3.test.csv",
    output:
        "results/smn1.corrs.csv",
        "results/smn1.inferred_interaction_strength.csv",
        "results/dmsc.corrs.csv",
        "results/dmsc.inferred_interaction_strength.csv",
        "results/gb1.corrs.csv",
        "results/gb1.inferred_interaction_strength.csv",
        "results/fyn-sh3.corrs.csv",
        "results/fyn-sh3.inferred_interaction_strength.csv",
    shell:
        "python code/datasets/fit.py"

rule plot_figure3:
    input:
        "results/smn1.corrs.csv",
        "results/smn1.inferred_interaction_strength.csv",
        "results/dmsc.corrs.csv",
        "results/dmsc.inferred_interaction_strength.csv",
    output:
        "figures/figure3.png",
        "figures/figure3.svg",
    shell:
        "python code/figures/main/figure3.py"

rule plot_figureS1:
    input:
        "results/gb1.corrs.csv",
        "results/gb1.inferred_interaction_strength.csv",
        "results/fyn-sh3.corrs.csv",
        "results/fyn-sh3.inferred_interaction_strength.csv",
    output:
        "figures/figureS1.png",
        "figures/figureS1.svg",
    shell:
        "python code/figures/supp/figureS1.py"


# Intron dataset (Figure 4, Figure 5, Figure S2, Figure S3)
rule process_intron_data:
    input:
        "data/raw/intron.csv",
    output:
        "data/processed/intron.30C.csv",
        "data/processed/intron.37C.csv",
    shell:
        "python code/intron/deseq2.py"

rule split_intron_data:
    input:
        "data/processed/intron.30C.csv",
        "data/processed/intron.37C.csv",
    output:
        "data/processed/intron.30C.train.csv",
        "data/processed/intron.30C.test.csv",
        "data/processed/intron.37C.train.csv",
        "data/processed/intron.37C.test.csv",
    shell:
        "python code/intron/split_train_test.py"

rule fit_intron:
    input:
        "data/processed/intron.30C.train.csv",
    output:
        "results/intron.30C.ler.a.npy",
        "results/intron.30C.ler.lambda_U.npy",
        "results/intron.30C.interaction_strength.csv",
        "results/intron.30C.corrs.csv",
    shell:
        "python code/intron/fit.py"

rule predict_intron_landscape:
    input:
        "data/processed/intron.30C.train.csv",
        "results/intron.30C.ler.a.npy",
        "results/intron.30C.ler.lambda_U.npy",
    output:
        "results/intron.30C.ler.landscape.csv",
    shell:
        "python code/intron/predict.py"

rule predict_intron_test:
    input:
        "data/processed/intron.30C.train.csv",
        "data/processed/intron.30C.test.csv",
        "results/intron.30C.ler.a.npy",
        "results/intron.30C.ler.lambda_U.npy",
    output:
        "results/intron.30C.ler.pred.csv",
    shell:
        "python code/intron/predict_var.py"

rule intron_contrasts:
    input:
        "data/processed/intron.30C.csv",
        "results/intron.30C.ler.a.npy",
        "results/intron.30C.ler.lambda_U.npy",
    output:
        "results/intron.30C.ler.contrasts.csv",
    shell:
        "python code/intron/contrasts.py"

rule intron_calc_variance_components:
    input:
        "results/intron.30C.ler.landscape.csv",
    output:
        "results/intron.30C.ler.sites_variance_k.csv",
        "results/intron.30C.ler.sites_pairs_variance.csv",
    shell:
        "python code/intron/calc_variance_components.py"

rule intron_calc_visualization_source:
    input:
        "results/intron.30C.ler.landscape.csv",
    output:
        "results/intron.30C.edges.npz",
        "results/intron.30C.ler.map.mf_1.9.nodes.pq",
        "results/intron.30C.ler.map.mf_1.9.decay_rates.csv",
    shell:
        "python code/intron/calc_visualization.py"

rule intron_calc_visualization:
    input:
        "results/intron.30C.edges.npz",
        "results/intron.30C.ler.map.mf_1.9.nodes.pq",
        "results/intron.30C.ler.map.mf_1.9.decay_rates.csv",
    output:
        expand("results/intron.30C.ler.map.mf_{mf}.nodes.pq", mf=INTRON_MFS),
        expand("results/intron.30C.ler.map.mf_{mf}.decay_rates.csv", mf=INTRON_MFS),
    run:
        import shutil
        for mf in INTRON_MFS:
            shutil.copyfile(
                "results/intron.30C.ler.map.mf_1.9.nodes.pq",
                f"results/intron.30C.ler.map.mf_{mf}.nodes.pq",
            )
            shutil.copyfile(
                "results/intron.30C.ler.map.mf_1.9.decay_rates.csv",
                f"results/intron.30C.ler.map.mf_{mf}.decay_rates.csv",
            )

rule intron_calc_epistatic_coefficients:
    input:
        "results/intron.30C.ler.landscape.csv",
    output:
        "results/intron.30C.ler.epistatic_coefficients.csv",
    shell:
        "python code/intron/calc_epistatic_coeffs.py"

rule intron_calc_r2_curves:
    input:
        "data/processed/intron.30C.csv",
    output:
        "results/intron.30C.r2.csv",
    shell:
        "python code/intron/calc_r2_curves.py"

rule plot_figure4:
    input:
        "results/intron.30C.corrs.csv",
        "results/intron.30C.interaction_strength.csv",
        "results/intron.30C.ler.landscape.csv",
        "data/processed/intron.30C.train.csv",
        "results/intron.30C.ler.pred.csv",
        "data/processed/intron.30C.test.csv",
        "results/intron.30C.r2.csv",
        "results/intron.30C.ler.sites_variance_k.csv",
        "results/intron.30C.ler.sites_pairs_variance.csv",
    output:
        "figures/figure4.png",
        "figures/figure4.svg",
    shell:
        "python code/figures/main/figure4.py"

rule plot_figure5a:
    input:
        "results/intron.30C.ler.map.mf_1.6.nodes.pq",
        "results/intron.30C.edges.npz",
    output:
        "figures/figure5a.png",
        "figures/figure5a.svg",
    shell:
        "python code/figures/main/figure5/figure5a.py"

rule plot_figure5bch:
    input:
        "results/intron.30C.ler.contrasts.csv",
    output:
        "figures/figure5bch.png",
        "figures/figure5bch.svg",
    shell:
        "python code/figures/main/figure5/figure5bch.py"

rule plot_figure5defg:
    input:
        "results/intron.30C.ler.map.mf_1.6.nodes.pq",
        "results/intron.30C.edges.npz",
        "results/intron.30C.ler.epistatic_coefficients.csv",
    output:
        "figures/figure5defg.png",
        "figures/figure5defg.svg",
    shell:
        "python code/figures/main/figure5/figure5defg.py"

rule plot_figureS2:
    input:
        "results/intron.30C.edges.npz",
        expand("results/intron.30C.ler.map.mf_{mf}.nodes.pq", mf=INTRON_MFS),
    output:
        "figures/figureS2.png",
        "figures/figureS2.svg",
    shell:
        "python code/figures/supp/figureS2.py"

rule plot_figureS3:
    input:
        "results/intron.30C.ler.map.mf_1.6.nodes.pq",
        "results/intron.30C.edges.npz",
    output:
        "figures/figureS3.png",
        "figures/figureS3.svg",
    shell:
        "python code/figures/supp/figureS3.py"
