shell.prefix("source activate.sh ; source $(conda info --base)/etc/profile.d/conda.sh ; conda activate deltaU ; ")

INTRON_MFS = [0, 0.4, 0.8, 1.2, 1.6, 1.8]
SIMULATION_MODELS = ["ler", "ssVC"]
DATASETS = ["smn1", "dmsc", "gb1", "fyn-sh3", "intron.30C"]
CV_SPLIT_IDS = list(range(1, 31))
CV_MODELS = ["MEI", "VC", "CN", "LER", "Additive", "Pairwise"]

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
        "figures/figure5b.png",
        "figures/figure5c.png",
        "figures/figure5h.png",
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

rule figureS4:
    input:
        "figures/figureS4.png",

rule figureS5:
    input:
        "figures/figureS5.png",

rule figureS6:
    input:
        "figures/figureS6.png",

rule figureS7:
    input:
        "figures/figureS7.png",

rule figureS8:
    input:
        "figures/figureS8.png",

rule figureS9:
    input:
        "figures/figureS9.png",

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
        rules.figureS4.input,
        rules.figureS5.input,
        rules.figureS6.input,
        rules.figureS7.input,
        rules.figureS8.input,
        rules.figureS9.input,

rule all:
    input:
        rules.main_figures.input,
        rules.supplementary_figures.input


# Simulations (Figure 2)
rule simulate_data:
    output:
        "results/simulations.ler.prior_a.csv",
        "results/simulations.ler.prior_correlations.csv",
        "results/simulations.ler.lambda_U.csv",
        "data/processed/simulations.ler.csv",
        "results/simulations.ssVC.prior_correlations.csv",
        "results/simulations.ssVC.lambda_U.csv",
        "data/processed/simulations.ssVC.csv",
    shell:
        "python code/simulations/simulate.py"

rule split_simulated_data:
    input:
        "data/processed/simulations.ler.csv",
        "data/processed/simulations.ssVC.csv",
    output:
        expand("data/processed/simulations.{model}.train.csv", model=SIMULATION_MODELS),
        expand("data/processed/simulations.{model}.test.csv", model=SIMULATION_MODELS),
    shell:
        "python code/simulations/split_train_test.py"

rule fit_simulated_data:
    input:
        expand("data/processed/simulations.{model}.train.csv", model=SIMULATION_MODELS),
        expand("data/processed/simulations.{model}.test.csv", model=SIMULATION_MODELS),
    output:
        expand("results/simulations.{model}.corrs.csv", model=SIMULATION_MODELS),
        expand("results/simulations.{model}.inferred_interaction_strength.csv", model=SIMULATION_MODELS),
        expand("results/simulations.{model}.inferred_lambda_U.ler.csv", model=SIMULATION_MODELS),
        expand("results/simulations.{model}.inferred_lambda_U.ssVC.csv", model=SIMULATION_MODELS),
        expand("results/simulations.{model}.pred.ler.csv", model=SIMULATION_MODELS),
        expand("results/simulations.{model}.pred.ssVC.csv", model=SIMULATION_MODELS),
    shell:
        "python code/simulations/fit.py"

rule calc_simulation_r2:
    input:
        "data/processed/simulations.ler.csv",
        "data/processed/simulations.ssVC.csv",
    output:
        "results/simulations.ler.r2.csv",
        "results/simulations.ssVC.r2.csv",
    shell:
        "python code/simulations/calc_r2_curves.py"

rule plot_figure2:
    input:
        "results/simulations.ler.prior_a.csv",
        "results/simulations.ler.prior_correlations.csv",
        "results/simulations.ler.corrs.csv",
        "results/simulations.ler.inferred_interaction_strength.csv",
        "results/simulations.ler.pred.ler.csv",
        "results/simulations.ler.r2.csv",
        "results/simulations.ssVC.r2.csv",
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

# rule split_cv_data:
#     input:
#         "data/processed/fyn-sh3.csv",
#         "data/processed/intron.30C.csv",
#     output:
#         expand("data/processed/splits/{dataset}.splits.csv", dataset=DATASETS),
#         expand("data/processed/splits/{dataset}.{i}.train.csv", dataset=DATASETS, i=CV_SPLIT_IDS),
#         expand("data/processed/splits/{dataset}.{i}.test.csv", dataset=DATASETS, i=CV_SPLIT_IDS),
#     shell:
#         "python code/datasets/split_cv_data.py"

# rule calc_datasets_r2_curves:
#     input:
#         rules.split_cv_data.output,
#     output:
#         expand("data/processed/splits/{dataset}.{i}.{model}.json",
#                dataset=DATASETS, i=CV_SPLIT_IDS, model=CV_MODELS),
#     shell:
#         "python code/datasets/calc_r2_curves.py"

# rule merge_datasets_r2_curves:
#     input:
#         rules.calc_datasets_r2_curves.output,
#     output:
#         expand("results/{dataset}.r2_curves.csv", dataset=DATASETS),
#     shell:
#         "python code/datasets/merge_cv_data.py"

rule plot_figure3:
    input:
        "results/smn1.corrs.csv",
        "results/smn1.inferred_interaction_strength.csv",
        "results/smn1.r2_curves.csv",
        "results/dmsc.corrs.csv",
        "results/dmsc.inferred_interaction_strength.csv",
        "results/dmsc.r2_curves.csv",
    output:
        "figures/figure3.png",
        "figures/figure3.svg",
    shell:
        "python code/figures/main/figure3.py"

rule plot_figureS1:
    input:
        "results/simulations.ler.lambda_U.csv",
        "results/simulations.ssVC.lambda_U.csv",
        "results/simulations.ler.inferred_lambda_U.ler.csv",
        "results/simulations.ler.inferred_lambda_U.ssVC.csv",
        "results/simulations.ssVC.inferred_lambda_U.ler.csv",
        "results/simulations.ssVC.inferred_lambda_U.ssVC.csv",
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
        "results/intron.30C.ssVC.lambda_U.npy",
        "results/intron.30C.interaction_strength.csv",
        "results/intron.30C.ler.corrs.csv",
    shell:
        "python code/intron/fit.py"

rule predict_intron_landscape:
    input:
        "data/processed/intron.30C.train.csv",
        "results/intron.30C.ler.a.npy",
        "results/intron.30C.ler.lambda_U.npy",
        "results/intron.30C.ssVC.lambda_U.npy",
    output:
        "results/intron.30C.ler.landscape.csv",
        "results/intron.30C.ssVC.landscape.csv",
    shell:
        "python code/intron/predict.py"

rule predict_intron_test:
    input:
        "data/processed/intron.30C.train.csv",
        "data/processed/intron.30C.test.csv",
        "results/intron.30C.ler.a.npy",
        "results/intron.30C.ler.lambda_U.npy",
        "results/intron.30C.ssVC.lambda_U.npy",
    output:
        "results/intron.30C.ler.pred.csv",
        "results/intron.30C.ssVC.pred.csv",
    shell:
        "python code/intron/predict_var.py"

rule intron_contrasts:
    input:
        "data/processed/intron.30C.csv",
        "results/intron.30C.ssVC.lambda_U.npy",
    output:
        "results/intron.30C.ssVC.contrasts.csv",
    shell:
        "python code/intron/calc_contrasts.py"

rule intron_calc_variance_components:
    input:
        "results/intron.30C.ssVC.landscape.csv",
    output:
        "results/intron.30C.ssVC.variance_k.csv",
        "results/intron.30C.ssVC.sites_variance_k.csv",
        "results/intron.30C.ssVC.sites_pairs_variance.csv",
        "results/intron.30C.ssVC.rmsec.csv",
    shell:
        "python code/intron/calc_variance_components.py"

rule intron_calc_visualization:
    input:
        "results/intron.30C.ssVC.landscape.csv",
    output:
        "results/intron.30C.edges.npz",
        expand("results/intron.30C.ssVC.map.mf_{mf}.nodes.pq", mf=INTRON_MFS),
        expand("results/intron.30C.ssVC.map.mf_{mf}.decay_rates.csv", mf=INTRON_MFS),
    shell:
        "python code/intron/calc_visualization.py"

rule intron_calc_epistatic_coefficients:
    input:
        "results/intron.30C.ssVC.landscape.csv",
    output:
        "results/intron.30C.ssVC.epistatic_coeffs.csv",
    shell:
        "python code/intron/calc_epistatic_coeffs.py"

rule intron_calc_mut_effs:
    input:
        "results/intron.30C.ssVC.landscape.csv",
    output:
        "results/intron.30C.ssVC.mut_effs.csv",
    shell:
        "python code/intron/calc_mut_effs_all.py"

rule intron_calc_gamma_statistics:
    input:
        "results/intron.30C.ssVC.landscape.csv",
    output:
        "results/intron.30C.ssVC.gamma_i_to_j.csv",
        "results/intron.30C.ssVC.gamma_i_to_jk.csv",
        "results/intron.30C.ssVC.gamma_UD_pairs.csv",
    shell:
        "python code/intron/calc_gamma_statistics.py"

rule intron_calc_gamma_allele_statistics:
    input:
        "results/intron.30C.ssVC.landscape.csv",
    output:
        "results/intron.30C.ssVC.gamma_AiBi_to_AjBj.csv",
        "results/intron.30C.ssVC.gamma_AiBiAjBj_D.csv",
    shell:
        "python code/intron/calc_gamma_allele_statistics.py"

rule plot_figure4:
    input:
        "data/processed/intron.30C.train.csv",
        "data/processed/intron.30C.test.csv",
        "results/intron.30C.ler.corrs.csv",
        "results/intron.30C.interaction_strength.csv",
        "results/intron.30C.ssVC.landscape.csv",
        "results/intron.30C.r2_curves.csv",
        "results/intron.30C.ssVC.pred.csv",
        "results/intron.30C.ssVC.rmsec.csv",
        "results/intron.30C.ssVC.sites_variance_k.csv",
        "results/intron.30C.ssVC.sites_pairs_variance.csv",
    output:
        "figures/figure4.png",
        "figures/figure4.svg",
    shell:
        "python code/figures/main/figure4.py"

rule plot_figure5a:
    input:
        "results/intron.30C.ssVC.map.mf_1.8.nodes.pq",
        "results/intron.30C.edges.npz",
    output:
        "figures/figure5a.png",
        "figures/figure5a.svg",
    shell:
        "python code/figures/main/figure5/figure5a.py"

rule plot_figure5bch:
    input:
        "results/intron.30C.ssVC.contrasts.csv",
    output:
        "figures/figure5b.png",
        "figures/figure5b.svg",
        "figures/figure5c.png",
        "figures/figure5c.svg",
        "figures/figure5h.png",
        "figures/figure5h.svg",
    shell:
        "python code/figures/main/figure5/figure5bch.py"

rule plot_figure5defg:
    input:
        "results/intron.30C.ssVC.map.mf_1.6.nodes.pq",
        "results/intron.30C.edges.npz",
        "results/intron.30C.ssVC.epistatic_coeffs.csv",
    output:
        "figures/figure5defg.png",
        "figures/figure5defg.svg",
    shell:
        "python code/figures/main/figure5/figure5defg.py"

rule plot_figureS2:
    input:
        "results/gb1.corrs.csv",
        "results/gb1.inferred_interaction_strength.csv",
        "results/gb1.r2_curves.csv",
        "results/fyn-sh3.corrs.csv",
        "results/fyn-sh3.inferred_interaction_strength.csv",
        "results/fyn-sh3.r2_curves.csv",
    output:
        "figures/figureS2.png",
        "figures/figureS2.svg",
    shell:
        "python code/figures/supp/figureS2.py"

rule plot_figureS3:
    input:
        "results/intron.30C.ssVC.gamma_i_to_j.csv",
        "results/intron.30C.ssVC.gamma_i_to_jk.csv",
        "results/intron.30C.ssVC.gamma_UD_pairs.csv",
    output:
        "figures/figureS3.png",
        "figures/figureS3.svg",
    shell:
        "python code/figures/supp/figureS3.py"

rule plot_figureS4:
    input:
        "results/intron.30C.ssVC.gamma_AiBi_to_AjBj.csv",
        "results/intron.30C.ssVC.gamma_AiBiAjBj_D.csv",
    output:
        "figures/figureS4.png",
        "figures/figureS4.svg",
    shell:
        "python code/figures/supp/figureS4.py"

rule plot_figureS5:
    input:
        "results/intron.30C.ssVC.mut_effs.csv",
        "results/intron.30C.ssVC.epistatic_coeffs.csv",
    output:
        "figures/figureS5.png",
    shell:
        "python code/figures/supp/figureS5.py"

rule plot_figureS6_7:
    input:
        "results/intron.30C.ssVC.mut_effs.csv",
        "results/intron.30C.ssVC.epistatic_coeffs.csv",
    output:
        "figures/figureS6.png",
        "figures/figureS7.png",
    shell:
        "python code/figures/supp/figureS6_7.py"

rule plot_figureS8:
    input:
        "results/intron.30C.edges.npz",
        expand("results/intron.30C.ssVC.map.mf_{mf}.nodes.pq", mf=INTRON_MFS),
    output:
        "figures/figureS8.png",
        "figures/figureS8.svg",
    shell:
        "python code/figures/supp/figureS8.py"

rule plot_figureS9:
    input:
        "results/intron.30C.edges.npz",
        "results/intron.30C.ssVC.map.mf_1.6.nodes.pq",
    output:
        "figures/figureS9.png",
        "figures/figureS9.svg",
    shell:
        "python code/figures/supp/figureS9.py"
