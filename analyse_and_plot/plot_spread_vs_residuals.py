"""Plot the absolute error vs the predicted spread"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

color_PaiNN = "#ea7317"
color_GPR_both = "#55a630"
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


residual_GPR_traj = np.abs(Y_predict_GPR_traj_per_seed[0].flatten() - Y_true.flatten())
sigmas_GPR_traj = sigma_GPR_traj_per_seed[0].flatten()

residuals_PaiNN = np.abs(Y_predict_PaiNN_per_seed[0].flatten() - Y_true.flatten())
sigmas_PaiNN = (
    np.sqrt(9 / (9 - 2)) * sigmas_PaiNN_per_seed[0].flatten()
)  # Std student's t-distrib. from sample std

residual_GPR_both = np.abs(Y_predict_GPR_both_per_seed[0].flatten() - Y_true.flatten())
sigmas_GPR_both = sigma_GPR_both_per_seed[0].flatten()

# Plot residuals vs standard deviation
plot_properties = {
    "font.size": 26,
    "xtick.major.size": 15,
    "ytick.major.size": 15,
    "xtick.major.width": 2,
    "ytick.major.width": 2,
    "axes.linewidth": 2,
    "legend.fontsize": 24,
}
mpl.rcParams.update(plot_properties)
fig, ax = plt.subplots(figsize=(10, 10))

ax.grid(True, alpha=0.2)

ax.scatter(
    residuals_PaiNN,
    sigmas_PaiNN,
    s=20,
    marker="o",
    alpha=0.7,
    linewidths=0.5,
    color=color_PaiNN,
    edgecolors="black",
    label=r"$\mathregular{PaiNN_{Ens}}$",
    zorder=3,
)

ax.scatter(
    residual_GPR_traj,
    sigmas_GPR_traj,
    s=20,
    marker="o",
    alpha=0.7,
    linewidths=0.5,
    color=color_GPR_traj,
    edgecolors="black",
    label=r"$\mathregular{SOAP_{Traj}}$",
    zorder=2,
)

ax.scatter(
    residual_GPR_both,
    sigmas_GPR_both,
    s=20,
    marker="o",
    alpha=0.4,
    linewidths=0.5,
    color=color_GPR_both,
    edgecolors="black",
    label=r"$\mathregular{SOAP_{Full}}$",
    zorder=1,
)

ax.legend(loc="upper right", ncol=1, borderaxespad=0, frameon=False, markerscale=3)

ax.set_xlabel(r"$|\Delta E^{predict}_i-\Delta E^{true}_i|$ [kcal/mol]")
ax.set_ylabel("Standard deviation [kcal/mol]")

ax.set_title("Spread vs Residuals")
ax.set_ylim(ymin=0)
plt.tight_layout()
plt.savefig(cwd / "figures" / "spread_vs_residuals.png", dpi=400, bbox_inches="tight")
plt.show()
