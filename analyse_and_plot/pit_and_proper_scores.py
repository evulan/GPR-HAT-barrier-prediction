"""Plots the probability integral transform and computes the CRPS and LogS"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
from scipy import stats
from sklearn.metrics import mean_absolute_error
from scipy.special import beta

color_PaiNN = "#ea7317"
color_GPR_both = "#06aed5"
color_GPR_traj = "#086788"

cwd = Path(__file__).resolve().parent

# Load the results
lookup_PaiNN = pd.read_pickle(cwd / "results" / "lookup_PaiNN.pkl")
lookup_GPR_both = pd.read_pickle(cwd / "results" / "lookup_GPR_both.pkl")
lookup_GPR_traj = pd.read_pickle(cwd / "results" / "lookup_GPR_traj.pkl")
df = pd.read_pickle(cwd / "results" / "all_predictions.pkl")
mask = (df.origin == "traj").to_numpy()

eps = np.finfo(float).eps
Y_true = df["E_barrier"].to_numpy().copy()[mask]

# PaiNN ensemble
seeds = np.unique(lookup_PaiNN["sample_seed"])
ranks = np.unique(lookup_PaiNN["rank"])
fracs_PaiNN = np.array([1.0, 0.05])
lookup_PaiNN = lookup_PaiNN[
    np.isclose(lookup_PaiNN["frac"], fracs_PaiNN[0], atol=1e-6)
    | np.isclose(lookup_PaiNN["frac"], fracs_PaiNN[1], atol=1e-6)
]

predictions_PaiNN = np.zeros((fracs_PaiNN.size, seeds.size, ranks.size, Y_true.size))
for i_frac, frac in enumerate(fracs_PaiNN):
    for i_seed, seed in enumerate(seeds):
        for i_rank, rank in enumerate(ranks):
            run_hash = lookup_PaiNN[
                (lookup_PaiNN["frac"] == frac)
                & (lookup_PaiNN["rank"] == rank)
                & (lookup_PaiNN["sample_seed"] == seed)
            ]
            assert run_hash.shape[0] == 1
            E_predict = df[f"E_PaiNN={run_hash.iloc[0].name}"].to_numpy()[mask]
            print(f"Seed: {seed}, Rank: {rank}", end="\r")
            predictions_PaiNN[i_frac, i_seed, i_rank] = E_predict

Y_predict_PaiNN_per_seed = np.mean(predictions_PaiNN, axis=2)
sigmas_PaiNN_per_seed = np.std(
    predictions_PaiNN, axis=2, ddof=1
)  # Bessel's correction, since only 10 models


# GPR train traj & synth
fracs_GPR_both = np.array([1.0, 0.05])
lookup_GPR_both = lookup_GPR_both[
    np.isclose(lookup_GPR_both["frac"], fracs_GPR_both[0], atol=1e-6)
    | np.isclose(lookup_GPR_both["frac"], fracs_GPR_both[1], atol=1e-6)
]
assert np.array_equal(seeds, np.unique(lookup_GPR_both["seed"]))

Y_predict_GPR_both_per_seed = np.zeros((fracs_GPR_both.size, seeds.size, Y_true.size))
sigma_GPR_both_per_seed = np.zeros((fracs_GPR_both.size, seeds.size, Y_true.size))
for i_frac, frac in enumerate(fracs_GPR_both):
    for i_seed, seed in enumerate(seeds):
        run_hash = lookup_GPR_both[
            (lookup_GPR_both["frac"] == frac) & (lookup_GPR_both["seed"] == seed)
        ]
        assert run_hash.shape[0] == 1
        E_predict = df[f"E_GPR_both={run_hash.iloc[0].name}"].to_numpy()[mask]
        Sigma_predict = df[f"Sigma_GPR_both={run_hash.iloc[0].name}"].to_numpy()[mask]
        print(f"Seed: {seed}", end="\r")
        Y_predict_GPR_both_per_seed[i_frac, i_seed] = E_predict
        sigma_GPR_both_per_seed[i_frac, i_seed] = Sigma_predict


# GPR train traj only
fracs_GPR_traj = np.array([0.5427, 0.05])
lookup_GPR_traj = lookup_GPR_traj[
    np.isclose(lookup_GPR_traj["frac"], fracs_GPR_traj[0], atol=1e-6)
    | np.isclose(lookup_GPR_traj["frac"], fracs_GPR_traj[1], atol=1e-6)
]
assert np.array_equal(seeds, np.unique(lookup_GPR_traj["seed"]))
Y_predict_GPR_traj_per_seed = np.zeros((fracs_GPR_traj.size, seeds.size, Y_true.size))
sigma_GPR_traj_per_seed = np.zeros((fracs_GPR_traj.size, seeds.size, Y_true.size))
for i_frac, frac in enumerate(fracs_GPR_traj):
    for i_seed, seed in enumerate(seeds):
        run_hash = lookup_GPR_traj[
            (lookup_GPR_traj["frac"] == frac) & (lookup_GPR_traj["seed"] == seed)
        ]
        assert run_hash.shape[0] == 1
        E_predict = df[f"E_GPR_traj={run_hash.iloc[0].name}"].to_numpy()[mask]
        Sigma_predict = df[f"Sigma_GPR_traj={run_hash.iloc[0].name}"].to_numpy()[mask]
        print(f"Seed: {seed}", end="\r")
        Y_predict_GPR_traj_per_seed[i_frac, i_seed] = E_predict
        sigma_GPR_traj_per_seed[i_frac, i_seed] = Sigma_predict


# Scoring rules for different distributions
def crps_gaussian(ys_true, means_predict, sigmas_predict):
    """(negatively oriented) CRPS score for Gaussian distribution"""
    ys_true = ys_true.flatten()
    means_predict = means_predict.flatten()
    sigmas_predict = sigmas_predict.flatten()

    phi = stats.norm.pdf
    PHI = stats.norm.cdf

    A = (ys_true - means_predict) / sigmas_predict
    crps_all = sigmas_predict * (
        A * (2 * PHI(A) - 1) + 2 * phi(A) - (1 / np.sqrt(np.pi))
    )
    return np.mean(crps_all)


def crps_tdistrib(ys_true, means_predict, sigmas_predict, nu):
    """(negatively oriented) CRPS score for Student’s t-distribution"""

    F_nu = lambda y: stats.t.cdf(y, df=nu, loc=0, scale=1)
    f_nu = lambda y: stats.t.pdf(y, df=nu, loc=0, scale=1)
    B = beta

    y = (ys_true.flatten() - means_predict.flatten()) / sigmas_predict.flatten()
    crps_all = (
        y * (2 * F_nu(y) - 1)
        + 2 * f_nu(y) * ((nu + y**2) / (nu - 1))
        - ((2 * np.sqrt(nu)) / (nu - 1)) * ((B(0.5, nu - 0.5)) / (B(0.5, nu / 2) ** 2))
    )
    crps_all = sigmas_predict * crps_all

    return np.mean(crps_all)


def logS_gaussian(ys_true, means_predict, sigmas_predict):
    """(negatively oriented) Logarithmic score for Gaussian distribution"""
    logS_all = -stats.norm.logpdf(
        ys_true.flatten(), loc=means_predict.flatten(), scale=sigmas_predict.flatten()
    )
    return np.mean(logS_all)


def logS_tdistrib(ys_true, means_predict, sigmas_predict, n):
    """(negatively oriented) Logarithmic score for Student’s t-distribution"""
    return -stats.t.logpdf(
        ys_true, df=n - 1, loc=means_predict, scale=sigmas_predict
    ).mean()


def PIT_gaussian(
    *,
    y_true,
    means_predict,
    sigmas_predict,
    ax,
    color,
    hatch=None,
    label="",
    linestyle="-",
):
    """Create PIT plot, when underlying distrib. is Gaussian distribution"""
    ys_true = y_true.flatten()
    means_predict = means_predict.flatten()
    sigmas_predict = sigmas_predict.flatten()

    p = stats.norm.cdf(ys_true, loc=means_predict, scale=sigmas_predict)
    ax.hist(
        x=p,
        bins=20,
        range=(0, 1),
        density=True,
        alpha=0.65,
        color=color,
        hatch=hatch,
        fill=True,
        edgecolor="black",
        linewidth=4,
        linestyle=linestyle,
        label=label,
        zorder=0,
    )


def PIT_tdistrib(
    *,
    y_true,
    means_predict,
    sigmas_predict,
    ax,
    color,
    hatch=None,
    label="",
    linestyle="-",
):
    """Create PIT plot, when underlying distrib. is student's t-distribution"""
    ys_true = y_true.flatten()
    means_predict = means_predict.flatten()
    sigmas_predict = sigmas_predict.flatten()

    p = stats.t.cdf(ys_true, loc=means_predict, scale=sigmas_predict, df=9)
    ax.hist(
        x=p,
        bins=20,
        range=(0, 1),
        density=True,
        alpha=0.65,
        color=color,
        hatch=hatch,
        fill=True,
        edgecolor="black",
        linewidth=4,
        linestyle=linestyle,
        label=label,
        zorder=0,
    )


# Compute scores
scores = np.zeros((2, fracs_PaiNN.size, 3))  # Scoring rule, frac, model
for i_frac in range(fracs_PaiNN.size):
    # CRPS
    CRPS_GPR_both = np.mean(
        [
            crps_gaussian(
                Y_true,
                Y_predict_GPR_both_per_seed[i_frac, seed_i],
                sigma_GPR_both_per_seed[i_frac, seed_i],
            )
            for seed_i in seeds
        ]
    )
    CRPS_GPR_traj = np.mean(
        [
            crps_gaussian(
                Y_true,
                Y_predict_GPR_traj_per_seed[i_frac, seed_i],
                sigma_GPR_traj_per_seed[i_frac, seed_i],
            )
            for seed_i in seeds
        ]
    )

    CRPS_PaiNN = np.mean(
        [
            crps_tdistrib(
                Y_true,
                Y_predict_PaiNN_per_seed[i_frac, seed_i],
                sigmas_PaiNN_per_seed[i_frac, seed_i],
                nu=9,
            )
            for seed_i in seeds
        ]
    )

    print(
        f"CRPS GPR Both: {CRPS_GPR_both:.2f} kcal/mol  (Frac: {fracs_GPR_both[i_frac]})"
    )
    print(
        f"CRPS GPR Traj: {CRPS_GPR_traj:.2f} kcal/mol (Frac: {fracs_GPR_traj[i_frac]})"
    )
    print(
        f"CRPS PaiNN Ensemble: {CRPS_PaiNN:.2f} kcal/mol (Frac: {fracs_PaiNN[i_frac]})"
    )

    # LogS
    LogS_GPR_both = np.mean(
        [
            logS_gaussian(
                Y_true,
                Y_predict_GPR_both_per_seed[i_frac, seed_i],
                sigma_GPR_both_per_seed[i_frac, seed_i],
            )
            for seed_i in seeds
        ]
    )
    LogS_GPR_traj = np.mean(
        [
            logS_gaussian(
                Y_true,
                Y_predict_GPR_traj_per_seed[i_frac, seed_i],
                sigma_GPR_traj_per_seed[i_frac, seed_i],
            )
            for seed_i in seeds
        ]
    )
    LogS_PaiNN = np.mean(
        [
            logS_tdistrib(
                Y_true,
                Y_predict_PaiNN_per_seed[i_frac, seed_i],
                sigmas_PaiNN_per_seed[i_frac, seed_i],
                n=10,
            )
            for seed_i in seeds
        ]
    )
    print(f"LogS GPR Both: {LogS_GPR_both:.2f} (Frac: {fracs_GPR_both[i_frac]})")
    print(f"LogS GPR Traj: {LogS_GPR_traj:.2f} (Frac: {fracs_GPR_traj[i_frac]})")
    print(f"LogS PaiNN Ensemble: {LogS_PaiNN:.2f} (Frac: {fracs_PaiNN[i_frac]})")

    scores[0, i_frac] = [CRPS_GPR_both, CRPS_GPR_traj, CRPS_PaiNN]
    scores[1, i_frac] = [LogS_GPR_both, LogS_GPR_traj, LogS_PaiNN]


# Print Latex table of scores
scores = pd.DataFrame(
    scores.reshape(-1, 3),
    index=[
        f"CRPS (All data)",
        f"CRPS (5% of all)",
        f"LogS (All data)",
        f"LogS (5% of all)",
    ],
    columns=[
        r"SOAP\textsubscript{Full}",
        r"SOAP\textsubscript{Traj}",
        r"PaiNN\textsubscript{Ens}",
    ],
)
print(scores.to_latex(index=True, float_format="{:.2f}".format))


# PIT plot
plot_properties = {
    "font.size": 26,
    "xtick.major.size": 15,
    "ytick.major.size": 15,
    "xtick.major.width": 2,
    "ytick.major.width": 2,
    "axes.linewidth": 2,
    "legend.fontsize": 26,
}
mpl.rcParams.update(plot_properties)
fig, ax = plt.subplots(1, 1, figsize=(12, 6))

y_true_all_seeds = np.tile(Y_true.flatten(), seeds.size)

PIT_gaussian(
    y_true=y_true_all_seeds,
    means_predict=Y_predict_GPR_traj_per_seed[0].flatten(),
    sigmas_predict=sigma_GPR_both_per_seed[0].flatten(),
    ax=ax,
    color=color_GPR_both,
    label=r"$\mathregular{SOAP_{Full}}$",
)
ax.hlines(1.0, *ax.get_xlim(), ls="--", color="black", lw=5, label="ideal", zorder=10)
PIT_gaussian(
    y_true=y_true_all_seeds,
    means_predict=Y_predict_GPR_both_per_seed[0].flatten(),
    sigmas_predict=sigma_GPR_traj_per_seed[0].flatten(),
    ax=ax,
    color=color_GPR_traj,
    hatch="o",
    linestyle=(0, (1, 1)),
    label=r"$\mathregular{SOAP_{Traj}}$",
)

PIT_tdistrib(
    y_true=y_true_all_seeds,
    means_predict=Y_predict_PaiNN_per_seed[0].flatten(),
    sigmas_predict=sigmas_PaiNN_per_seed[0].flatten(),
    ax=ax,
    color=color_PaiNN,
    label=r"$\mathregular{PaiNN_{Ens}}$",
)

ax.spines[["right", "top"]].set_visible(False)

ax.set_xlabel(r"Probability Integral Transform")
x_ticks = np.arange(0.0, 1.1, 0.1)
ax.set_xticks(x_ticks - 0.025)
ax.set_xticklabels([f"{t:.1f}" for t in x_ticks], rotation=0)
ax.set_xlim([0.0, 1])
ax.set_ylabel(r"Relative Frequency")

ax.legend(frameon=False, ncol=3)
plt.tight_layout()
plt.savefig(cwd / "figures" / "pit.png", pad_inches=0, bbox_inches="tight", dpi=400)
plt.show()
