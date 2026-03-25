from itertools import product

import matplotlib.pyplot as plt
import numpy as np
from gpmap.linop import KronOperator, ProjectionOperator
from gpmap.summary import GPmapSummarizer
from scipy.special import logsumexp

if __name__ == "__main__":
    np.random.seed(0)
    alphabet = list("ACGT")
    alpha = len(alphabet)
    l = 10
    n = alpha**l
    w = 4
    seqs = np.array(["".join(s) for s in product(alphabet, repeat=l)])
    

    print("Initializing model configuration")
    print(f"  Alphabet = {alphabet}")
    print(f"  Sequence length = {l}")
    print(f"  Number of sequences = {n}")
    print(f"  Window size = {w}")

    print("Sampling random pairwise landscape")
    lambdas = np.zeros(l+1)
    lambdas[1:3] = 1
    P = ProjectionOperator(alpha, l, lambdas=lambdas)
    z = np.random.normal(size=n)
    phi = P @ z

    print("Maginalizing over the first 4 positions")
    I = np.eye(alpha)
    J = np.ones((1, alpha))
    matrices = [I] * w + [J] * (l - w)
    M = KronOperator(matrices)

    loglambdas = np.arange(-3, 5)
    lambdas = 10. ** loglambdas
    vcs = []
    vcs_perc = []
    for lambda_2 in lambdas:
        logp = -lambda_2 * phi - logsumexp(-lambda_2 * phi)
        p = np.exp(logp)
        marginal_p = M @ p
        marginal_logp = np.log(marginal_p)
        s = GPmapSummarizer(alpha, w, marginal_logp)
        vc = s.calc_V_k_variance_components()
        vcs.append(vc["variance"].values)
        vcs_perc.append(vc["variance_perc"].values)
    vcs = np.vstack(vcs).T
    vcs_perc = np.vstack(vcs_perc).T
    print(vcs)
    print(vcs_perc)

    print("Plotting variance components over the marginalized landscape")
    fig, subplots = plt.subplots(2, 1, figsize=(4, 3.75))
    axes = subplots[0]
    im = axes.imshow(np.log10(vcs), cmap="binary")
    axes.set(
        xlabel=r"$\log_{10}\lambda_2$",
        xticks=np.arange(lambdas.shape[0]),
        xticklabels=loglambdas,
        ylabel="Interaction order",
        yticks=np.arange(vcs.shape[0]),
        yticklabels=1 + np.arange(vcs.shape[0]),
    )
    plt.colorbar(im, label=r"$\log_{10}$(variance)")
    axes = subplots[1]
    im = axes.imshow(vcs_perc, cmap="binary", vmin=0, vmax=100)
    axes.set(
        xlabel=r"$\log_{10}\lambda_2$",
        xticks=np.arange(lambdas.shape[0]),
        xticklabels=loglambdas,
        ylabel="Interaction order",
        yticks=np.arange(vcs.shape[0]),
        yticklabels=1 + np.arange(vcs.shape[0]),
    )
    plt.colorbar(im, label="% variance explained")
    fig.tight_layout()
    fig.savefig('figures/yiyun_induced_variances.png', dpi=300)
    print('Done.')