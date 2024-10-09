"""Plot the prediction interval vs empirical coverage"""

import pandas as pd
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
from pathlib import Path

color_PaiNN = "#ea7317"
color_GPR_both = "#06aed5"
color_GPR_traj = "#086788"

cwd = Path(__file__).resolve().parent

# Load predictions
lookup_PaiNN = pd.read_pickle(cwd / "results" / "lookup_PaiNN.pkl")
lookup_GPR_both = pd.read_pickle(cwd / "results" / "lookup_GPR_both.pkl")
lookup_GPR_traj = pd.read_pickle(cwd / "results" / "lookup_GPR_traj.pkl")
df = pd.read_pickle(cwd / "results" / "all_predictions.pkl")
mask = (df.origin == "traj").to_numpy()

eps = np.finfo(float).eps
Y_true = df["E_barrier"].to_numpy().copy()[mask]

# PaiNN
lookup_PaiNN = lookup_PaiNN[np.isclose(lookup_PaiNN["frac"], 1.0, atol=1e-6)]
seeds = np.unique(lookup_PaiNN["sample_seed"])
ranks = np.unique(lookup_PaiNN["rank"])

predictions_PaiNN = np.zeros((seeds.size, ranks.size, Y_true.size))
for i_seed, seed in enumerate(seeds):
    for i_rank, rank in enumerate(ranks):
        run_hash = lookup_PaiNN[
            (lookup_PaiNN["rank"] == rank) & (lookup_PaiNN["sample_seed"] == seed)
        ]
        assert run_hash.shape[0] == 1
        E_predict = df[f"E_PaiNN={run_hash.iloc[0].name}"].to_numpy()[mask]
        print(f"Seed: {seed}, Rank: {rank}", end="\r")
        predictions_PaiNN[i_seed, i_rank] = E_predict

Y_predict_PaiNN_per_seed = np.mean(predictions_PaiNN, axis=1)
sigmas_PaiNN_per_seed = np.std(
    predictions_PaiNN, axis=1, ddof=1
)  # Bessel's correction, since only 10 models


# GPR SOAP (traj & synth)
lookup_GPR_both = lookup_GPR_both[np.isclose(lookup_GPR_both["frac"], 1.0, atol=1e-6)]
assert np.array_equal(seeds, np.unique(lookup_GPR_both["seed"]))

Y_predict_GPR_both_per_seed = np.zeros((seeds.size, Y_true.size))
sigma_GPR_both_per_seed = np.zeros((seeds.size, Y_true.size))
for i_seed, seed in enumerate(seeds):
    run_hash = lookup_GPR_both[lookup_GPR_both["seed"] == seed]
    assert run_hash.shape[0] == 1
    E_predict = df[f"E_GPR_both={run_hash.iloc[0].name}"].to_numpy()[mask]
    Sigma_predict = df[f"Sigma_GPR_both={run_hash.iloc[0].name}"].to_numpy()[mask]
    print(f"Seed: {seed}", end="\r")
    Y_predict_GPR_both_per_seed[i_seed] = E_predict
    sigma_GPR_both_per_seed[i_seed] = Sigma_predict


# GPR (traj only)
lookup_GPR_traj = lookup_GPR_traj[
    np.isclose(lookup_GPR_traj["frac"], 0.5427, atol=1e-6)
]
assert np.array_equal(seeds, np.unique(lookup_GPR_traj["seed"]))
Y_predict_GPR_traj_per_seed = np.zeros((seeds.size, Y_true.size))
sigma_GPR_traj_per_seed = np.zeros((seeds.size, Y_true.size))
for i_seed, seed in enumerate(seeds):
    run_hash = lookup_GPR_traj[lookup_GPR_traj["seed"] == seed]
    assert run_hash.shape[0] == 1
    E_predict = df[f"E_GPR_traj={run_hash.iloc[0].name}"].to_numpy()[mask]
    Sigma_predict = df[f"Sigma_GPR_traj={run_hash.iloc[0].name}"].to_numpy()[mask]
    print(f"Seed: {seed}", end="\r")
    Y_predict_GPR_traj_per_seed[i_seed] = E_predict
    sigma_GPR_traj_per_seed[i_seed] = Sigma_predict


def get_bounds_gaussian(alpha, means, sigmas):
    """Get Gaussian quantile bounds"""
    return stats.norm.ppf(alpha / 2, loc=means, scale=sigmas), stats.norm.ppf(
        1 - alpha / 2, loc=means, scale=sigmas
    )


def get_bounds_tdistrib(alpha, means, sigmas, df):
    """Get student's t-distribution bounds"""
    return stats.t.ppf(alpha / 2, df=df, loc=means, scale=sigmas), stats.t.ppf(
        1 - alpha / 2, df=df, loc=means, scale=sigmas
    )


def interval_score(*, alpha, Y_true, Y_predict, l, u):
    """Calculate the negatively oriented interval score"""
    true = Y_true.flatten()
    mean = Y_predict.flatten()
    assert true.size == mean.size == l.shape[0] == u.shape[0]
    n_above = np.sum(true > u)
    n_below = np.sum(true < l)
    n_total = mean.size
    within_bounds = 1 - (n_above + n_below) / n_total
    mean_interval = np.mean(u - l)
    S = (
        (u - l)
        + (2 / alpha) * (l - true) * (true < l)
        + (2 / alpha) * (true - u) * (true > u)
    )
    S_mean = np.mean(S)
    return S, S_mean, mean_interval, within_bounds


def plot_empirical_coverage(
    Y_true, Y_predict, sigma, mode, color="#FF5722", label="", ax=None
):
    """Plot the empirical coverage vs the predicted coverage"""
    alphas = np.linspace(eps, 1 - eps, 1000)
    prediction_coverage = 1 - alphas

    empirical_coverage = np.zeros(prediction_coverage.shape)
    for i, alpha in enumerate(alphas):

        if mode == "gaussian":
            l, u = get_bounds_gaussian(alpha, Y_predict, sigma)
        elif mode == "tdistrib":
            l, u = get_bounds_tdistrib(alpha, Y_predict, sigma, df=10)
        empirical_coverage[i] = interval_score(
            alpha=alpha, Y_true=Y_true, Y_predict=Y_predict, l=l, u=u
        )[-1]

    if ax is None:
        plot_properties = {
            "font.size": 35,
            "xtick.major.size": 15,
            "ytick.major.size": 15,
            "xtick.major.width": 2,
            "ytick.major.width": 2,
            "axes.linewidth": 2,
            "legend.fontsize": 26,
        }
        mpl.rcParams.update(plot_properties)
        fig, ax = plt.subplots(figsize=(14, 10))

    ax.plot(
        100 * prediction_coverage,
        100 * empirical_coverage,
        label=label,
        color=color,
        lw=2,
        alpha=0.5,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.grid(True, alpha=0.2)

    ax.set_ylabel(r"Empirical Coverage $[\%]$")
    ax.set_ylim(0, 100)
    ax.set_xlim(0, 100)
    ax.set_xticks(np.arange(0, 101, 10))
    ax.set_yticks(np.arange(0, 101, 10))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=-45)

    ax.set_aspect("equal", "box")
    return ax


# Plot
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
fig, ax = plt.subplots(1, 1, figsize=(12, 12), sharex=False)

for seed_i, seed in enumerate(seeds):
    plot_empirical_coverage(
        Y_true,
        Y_predict_PaiNN_per_seed[seed_i],
        sigma=sigmas_PaiNN_per_seed[seed_i],
        mode="tdistrib",
        color=color_PaiNN,
        label=rf"PaiNN$^{{[{seed}]}}$",
        ax=ax,
    )
    plot_empirical_coverage(
        Y_true,
        Y_predict_GPR_both_per_seed[seed_i],
        sigma=sigma_GPR_both_per_seed[seed_i],
        mode="gaussian",
        color=color_GPR_both,
        label=rf"GPR both$^{{[{seed}]}}$",
        ax=ax,
    )
    plot_empirical_coverage(
        Y_true,
        Y_predict_GPR_traj_per_seed[seed_i],
        sigma=sigma_GPR_traj_per_seed[seed_i],
        mode="gaussian",
        color=color_GPR_traj,
        label=rf"GPR both$^{{[{seed}]}}$",
        ax=ax,
    )

diag = np.linspace(*ax.get_xlim(), 1000)
ax.plot(diag, diag, color="black", ls="--", zorder=0, label="Ideal", lw=2)
ax.set_xlabel(r"Prediction Interval ($1-\alpha$) $[\%]$")
legend_items = []
legend_items.append(
    Line2D([0], [0], color="black", linewidth=2, linestyle="--", label="Ideal")
)
legend_items.append(
    Line2D(
        [0],
        [0],
        color=color_GPR_both,
        linewidth=2,
        linestyle="-",
        label=r"$\mathregular{SOAP_{Full}}$",
    )
)
legend_items.append(
    Line2D(
        [0],
        [0],
        color=color_GPR_traj,
        linewidth=2,
        linestyle="-",
        label=r"$\mathregular{SOAP_{Traj}}$",
    )
)
legend_items.append(
    Line2D(
        [0],
        [0],
        color=color_PaiNN,
        linewidth=2,
        linestyle="-",
        label=r"$\mathregular{PaiNN_{Ens}}$",
    )
)
ax.legend(
    handles=legend_items, loc="upper left", ncol=1, borderaxespad=0, frameon=False
)

plt.tight_layout()
# Save separately
fig.savefig(cwd / "figures" / "coverage.png", pad_inches=0, dpi=400)
plt.show()


# Interval score
def average_interval_score_table(
    Y_true, Y_predict_seeds_models, sigma_seeds_models, modes, model_names, n_seeds
):
    """Create a table of the mean intervals scores for different methods"""
    prediction_intervals = np.array([0.5, 0.8, 0.9, 0.95, 0.99])
    alphas = 1 - prediction_intervals
    S = np.zeros((alphas.size, len(model_names), n_seeds))
    for alpha_i, alpha in enumerate(alphas):
        for model_i in range(len(model_names)):
            for seed_i in range(n_seeds):
                Y_predict = Y_predict_seeds_models[model_i][seed_i]
                sigma = sigma_seeds_models[model_i][seed_i]
                if modes[model_i] == "gaussian":
                    l, u = get_bounds_gaussian(alpha, Y_predict, sigma)
                elif modes[model_i] == "tdistrib":
                    l, u = get_bounds_tdistrib(alpha, Y_predict, sigma, df=10)
                S[alpha_i][model_i][seed_i] = interval_score(
                    alpha=alpha, Y_true=Y_true, Y_predict=Y_predict, l=l, u=u
                )[1]
    S_means = np.mean(S, axis=-1)
    S_means_sigma = np.std(S, axis=-1, ddof=1)

    best_mask = np.full(S_means.shape, False)
    best_mask[np.arange(S_means.shape[0]), np.argmin(S_means, axis=1)] = True

    assert (
        S_means.shape[0] == len(alphas)
        and S_means.shape[1] == len(model_names)
        and S_means.ndim == 2
    )
    prediction_intervals_names = np.array(
        [f"{pi*100:.0f}\%" for pi in prediction_intervals]
    )
    index = pd.Index(prediction_intervals_names, name="PI")
    data = np.array(
        [
            # rf"\num{{{s[0]:.1f}\pm{s[1]:.1f}}}"  # with std
            rf"\num{{{s[0]:.1f}}}"
            for s in zip(S_means.flatten(), S_means_sigma.flatten())
        ],
        dtype="|U128",
    ).reshape(alphas.size, len(model_names))
    data[best_mask] = [rf"\textbf{{{d}}}" for d in data[best_mask]]
    df = pd.DataFrame(data, index=index, columns=model_names)
    print(
        df.to_latex(
            index=True,
            formatters={"PI": "{:.2f}".format},
            float_format="{:.1f}".format,
            escape=False,
        )
    )


# Collect inputs needed to create mean interval score table
Y_predict_models = [
    Y_predict_GPR_both_per_seed,
    Y_predict_GPR_traj_per_seed,
    Y_predict_PaiNN_per_seed,
]
sigma_models = [
    sigma_GPR_both_per_seed,
    sigma_GPR_traj_per_seed,
    sigmas_PaiNN_per_seed,
]
modes = ["gaussian", "gaussian", "tdistrib"]
model_names = [
    r"SOAP\textsubscript{Full}",
    r"SOAP\textsubscript{Traj}",
    r"PaiNN\textsubscript{Ens}",
]
average_interval_score_table(
    Y_true, Y_predict_models, sigma_models, modes, model_names, len(seeds)
)
