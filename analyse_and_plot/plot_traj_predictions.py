"""Plot the predictions for the trajectory test set for GPR SOAP, GPR MGK and PaiNN"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import matplotlib as mpl
from sklearn.metrics import mean_absolute_error
from scipy import stats

cwd = Path(__file__).resolve().parent
project_root = cwd.parent
results_folder = cwd / "results"

lookup_PaiNN = pd.read_pickle(results_folder / "lookup_PaiNN.pkl")
lookup_GPR_SOAP = pd.read_pickle(results_folder / "lookup_GPR_both.pkl")
df_SOAP_PaiNN = pd.read_pickle(results_folder / "all_predictions.pkl")
df_MGK = pd.read_pickle(project_root / "MGK_GPR" / "results" / "df_MGK_predictions.pkl")

seed_to_use = {"PaiNN": 0, "GPR_SOAP": 0}
save_fig_loc = cwd / "figures" / "compare_method_predictions.png"

assert np.all(np.isin(df_MGK.index.to_numpy(), df_SOAP_PaiNN.index.to_numpy()))
df_MGK = df_MGK.loc[df_SOAP_PaiNN.index].copy()

assert np.all(
    np.char.equal(
        df_MGK.index.to_numpy().astype(str), df_SOAP_PaiNN.index.to_numpy().astype(str)
    )
)
assert np.allclose(
    df_SOAP_PaiNN["E_barrier"].to_numpy().astype(float),
    df_MGK["E_barrier"].to_numpy().astype(float),
)
assert np.all(
    np.char.equal(
        df_MGK["origin"].to_numpy().astype(str),
        df_SOAP_PaiNN["origin"].to_numpy().astype(str),
    )
)
assert np.allclose(
    df_SOAP_PaiNN["d"].to_numpy().astype(float), df_MGK["d"].to_numpy().astype(float)
)

mask = (df_SOAP_PaiNN.origin == "traj").to_numpy()

eps = np.finfo(float).eps
Y_true = df_SOAP_PaiNN["E_barrier"].to_numpy().copy()[mask]

lookup_PaiNN = lookup_PaiNN[np.isclose(lookup_PaiNN["frac"], 1.0)]
seeds = np.unique(lookup_PaiNN["sample_seed"])
ranks = np.unique(lookup_PaiNN["rank"])

# PaiNN
predictions_PaiNN = np.zeros((seeds.size, ranks.size, Y_true.size))
for i_seed, seed in enumerate(seeds):
    for i_rank, rank in enumerate(ranks):
        run_hash = lookup_PaiNN[
            (lookup_PaiNN["rank"] == rank) & (lookup_PaiNN["sample_seed"] == seed)
        ]
        assert run_hash.shape[0] == 1
        E_predict = df_SOAP_PaiNN[f"E_PaiNN={run_hash.iloc[0].name}"].to_numpy()[mask]
        print(f"Seed: {seed}, Rank: {rank}", end="\r")
        predictions_PaiNN[i_seed, i_rank] = E_predict

Y_predict_PaiNN_per_seed = np.mean(predictions_PaiNN, axis=1)

Y_predict_PaiNN = Y_predict_PaiNN_per_seed[seed_to_use["PaiNN"]]
print(f"MAE PaiNN: {mean_absolute_error(Y_true, Y_predict_PaiNN):.2f} kcal/mol")

# GPR SOAP
lookup_GPR_SOAP = lookup_GPR_SOAP[np.isclose(lookup_GPR_SOAP["frac"], 1.0)]
assert np.array_equal(seeds, np.unique(lookup_GPR_SOAP["seed"]))

Y_predict_GPR_SOAP_per_seed = np.zeros((seeds.size, Y_true.size))
sigma_GPR_SOAP_per_seed = np.zeros((seeds.size, Y_true.size))
for i_seed, seed in enumerate(seeds):
    run_hash = lookup_GPR_SOAP[lookup_GPR_SOAP["seed"] == seed]
    assert run_hash.shape[0] == 1
    E_predict = df_SOAP_PaiNN[f"E_GPR_both={run_hash.iloc[0].name}"].to_numpy()[mask]
    Sigma_predict = df_SOAP_PaiNN[f"Sigma_GPR_both={run_hash.iloc[0].name}"].to_numpy()[
        mask
    ]
    print(f"Seed: {seed}", end="\r")
    Y_predict_GPR_SOAP_per_seed[i_seed] = E_predict
    sigma_GPR_SOAP_per_seed[i_seed] = Sigma_predict

Y_predict_GPR_SOAP = Y_predict_GPR_SOAP_per_seed[seed_to_use["GPR_SOAP"]]
print(f"MAE SOAP : {mean_absolute_error(Y_true, Y_predict_GPR_SOAP):.2f} kcal/mol")


# GPR marginalized_graph_kernel
Y_predict_GPR_MGK = df_MGK["E_barrier_predict"].to_numpy().astype(float)[mask]
print(f"MAE MGK  : {mean_absolute_error(Y_true, Y_predict_GPR_MGK):.2f} kcal/mol")


# Plot
color_PaiNN = "#ea7317"
color_GPR_SOAP = "#06aed5"
color_GPR_MGK = "#485696"

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
    Y_true,
    Y_predict_GPR_SOAP,
    s=30,
    marker="o",
    alpha=0.7,
    linewidths=0.5,
    color=color_GPR_SOAP,
    edgecolors="black",
    label=r"$\mathregular{SOAP_{Full}}$",
    zorder=3,
)
ax.scatter(
    Y_true,
    Y_predict_GPR_MGK,
    s=30,
    marker="o",
    alpha=0.7,
    linewidths=0.5,
    color=color_GPR_MGK,
    edgecolors="black",
    label="MGK",
    zorder=0,
)
ax.scatter(
    Y_true,
    Y_predict_PaiNN,
    s=30,
    marker="s",
    alpha=0.7,
    linewidths=0.5,
    color=color_PaiNN,
    edgecolors="black",
    label=r"$\mathregular{PaiNN_{Ens}}$",
    zorder=1,
)

ax.set_aspect("equal", "box")
ax.set_xlabel(r"$\Delta E^{True}$ [kcal/mol]")
ax.set_ylabel(r"$\Delta E^{Predicted}$ [kcal/mol]")

ax_min = np.min([Y_true, Y_predict_PaiNN, Y_predict_GPR_SOAP, Y_predict_GPR_MGK])
ax_max = np.max([Y_true, Y_predict_PaiNN, Y_predict_GPR_SOAP, Y_predict_GPR_MGK])

ax.set_xlim(20, 150)
ax.set_ylim(20, 150)

# Check that no point is outside limits
assert (
    ax_min >= ax.get_xlim()[0]
    and ax_max <= ax.get_xlim()[1]
    and ax_min >= ax.get_ylim()[0]
    and ax_max <= ax.get_ylim()[1]
)

ticks = np.arange(ax.set_xlim()[0], ax.set_xlim()[1] + 0.1, 10)
ax.set_xticks(ticks)
ax.set_xticklabels(ticks.astype(int), rotation=-45)
ax.set_yticks(ticks)
ax.set_yticklabels(ticks.astype(int), rotation=0)
diag = np.linspace(np.max([ax.get_xlim()[0], 0.0]), ax.get_xlim()[1], 1000)
ax.plot(diag, diag, color="black", ls="--", zorder=0, lw=2)
ax.legend(loc="upper left", ncol=1, borderaxespad=0, frameon=False, markerscale=3)

print(f"Save fig ({save_fig_loc.resolve()}) and show")
plt.tight_layout()
plt.savefig(save_fig_loc, dpi=400, bbox_inches="tight")
plt.show()


# Errors
fig, ax = plt.subplots(figsize=(10, 10))

ax.grid(True, alpha=0.2)

residuals_GPR_SOAP = np.abs(Y_true.flatten() - Y_predict_GPR_SOAP.flatten())
residuals_PaiNN = np.abs(Y_true.flatten() - Y_predict_PaiNN.flatten())
residuals_MGK = np.abs(Y_true.flatten() - Y_predict_GPR_MGK.flatten())

res_GPR_SOAP = stats.ecdf(residuals_GPR_SOAP)
res_GPR_PaiNN = stats.ecdf(residuals_PaiNN)
res_GPR_MGK = stats.ecdf(residuals_MGK)

ax = plt.subplot()
(line,) = res_GPR_SOAP.cdf.plot(ax)
line.set_label("GPR SOAP")
line.set_color(color_GPR_SOAP)

(line,) = res_GPR_PaiNN.cdf.plot(ax)
line.set_label("PaiNN Ensemble")
line.set_color(color_PaiNN)

(line,) = res_GPR_MGK.cdf.plot(ax)
line.set_label("MGK")
line.set_color(color_GPR_MGK)

ax.legend()
ax.set_xscale("log")
ax.set_xlabel(r"$|\Delta E^{predict}_i-\Delta E^{true}_i|$ [kcal/mol]")
ax.set_ylabel("Empirical CDF")
ax.set_title("Error distribution Trajectory Test Set")
plt.tight_layout()
plt.savefig(cwd / "figures" / "empirical_cdf_predictions", dpi=400, bbox_inches="tight")
plt.show()
