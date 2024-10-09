"""barrier_errors_vs_predicted_barriers"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import sys

sys.path.append(str(Path.cwd().parent.parent.resolve()))
sys.path.append(str(Path.cwd().parent.resolve()))

import matplotlib as mpl
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


def get_R2(y_true, y_pred):
    return 1 - (((y_true - y_pred) ** 2).sum() / ((y_true - y_true.mean()) ** 2).sum())


def get_corr(y_true, y_pred):
    numerator = ((y_true - y_true.mean()) * (y_pred - y_pred.mean())).sum()
    denominator = np.sqrt(
        ((y_true - y_true.mean()) ** 2).sum() * ((y_pred - y_pred.mean()) ** 2).sum()
    )
    return numerator / denominator


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
print(f"MAE  PaiNN: {mean_absolute_error(Y_true, Y_predict_PaiNN):.2f} kcal/mol")
print(f"RMSE PaiNN: {root_mean_squared_error(Y_true, Y_predict_PaiNN):.2f} kcal/mol")
print(f"R2   PaiNN: {get_R2(Y_true, Y_predict_PaiNN):.2f}")
print(f"corr PaiNN: {get_corr(Y_true, Y_predict_PaiNN):.2f}")


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
print(f"MAE  SOAP : {mean_absolute_error(Y_true, Y_predict_GPR_SOAP):.2f} kcal/mol")
print(f"RMSE SOAP: {root_mean_squared_error(Y_true, Y_predict_GPR_SOAP):.2f} kcal/mol")
print(f"R2   SOAP: {get_R2(Y_true, Y_predict_GPR_SOAP):.2f}")
print(f"corr SOAP: {get_corr(Y_true, Y_predict_GPR_SOAP):.2f}")

# GPR marginalized_graph_kernel
Y_predict_GPR_MGK = df_MGK["E_barrier_predict"].to_numpy().astype(float)[mask]
print(f"MAE  MGK  : {mean_absolute_error(Y_true, Y_predict_GPR_MGK):.2f} kcal/mol")
print(f"RMSE MGK: {root_mean_squared_error(Y_true, Y_predict_GPR_MGK):.2f} kcal/mol")
print(f"R2   MGK: {get_R2(Y_true, Y_predict_GPR_MGK):.2f}")
print(f"corr MGK: {get_corr(Y_true, Y_predict_GPR_MGK):.2f}")

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
fig, axes = plt.subplots(3, 1, figsize=(10, 18), sharex=True, sharey=True)

# Errors
df_pred = pd.DataFrame(
    {
        "Y_true": Y_true.flatten(),
        "Y_GPR": Y_predict_GPR_SOAP.flatten(),
        "Y_painn": Y_predict_PaiNN.flatten(),
        "Y_MGK": Y_predict_GPR_MGK.flatten(),
    }
)

colors = [color_GPR_SOAP, color_GPR_MGK, color_PaiNN]
pred_cols = ["Y_GPR", "Y_MGK", "Y_painn"]
titles = ["GPR SOAP", "GPR MGK", "GNN PaiNN"]

for i_axes, ax in enumerate(axes.flatten()):

    pred_col = pred_cols[i_axes]
    res_col = f"res_{pred_col}"
    res_col_mean = f"res_{pred_col}_mean"
    res_col_std = f"res_{pred_col}_std"
    df_pred.sort_values(pred_col, inplace=True)
    df_pred[res_col] = df_pred[pred_col] - df_pred["Y_true"]
    df_pred[res_col_std] = df_pred[res_col].rolling(window=30, center=True).std()
    df_pred[res_col_mean] = df_pred[res_col].rolling(window=30, center=True).mean()

    ax.scatter(df_pred[pred_col], df_pred[res_col], color=colors[i_axes], s=15)

    ax.fill_between(
        df_pred[pred_col],
        df_pred[res_col_mean] - df_pred[res_col_std],
        df_pred[res_col_mean] + df_pred[res_col_std],
        color="black",
        alpha=0.3,
    )  # , step="mid"
    ax.plot(df_pred[pred_col], df_pred[res_col_mean], "-", lw=3, color="black")

    if i_axes == axes.size - 1:
        ax.set_xlabel(r"$\Delta E_{\text{Predicted}}$")
    ax.set_ylabel(r"$\Delta E_{\text{Predicted}}-\Delta E_{\text{True}}$")
    ax.set_title(f"{titles[i_axes]}")

    ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig(cwd / "figures" / "residuals_spread.png", bbox_inches="tight", dpi=300)
plt.show()
