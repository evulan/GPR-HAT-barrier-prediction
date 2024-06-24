"""Plots the MAE of the optimized barriers vs distance cut-off"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import matplotlib as mpl

cwd = Path(__file__).resolve().parent
project_root = cwd.parent
results_folder = project_root / "two_stage_learning_PaiNN_SOAP" / "results"

# Load the SOAP+PaiNN predictions
painn_seeds = np.sort(
    [
        int(res_file.stem.split("df_opt_predictions_")[-1])
        for res_file in results_folder.glob(f"df_opt_predictions_*.pkl")
    ]
)
dfs_opt_test = [
    pd.read_pickle(results_folder / f"df_opt_predictions_{s}.pkl").set_index(
        "hash_direction"
    )
    for s in painn_seeds
]

d = dfs_opt_test[0]["d"].to_numpy().astype(float)
origin = dfs_opt_test[0]["origin"].to_numpy().astype(str)
Y_predict_per_seed = np.array([df["E_barrier_opt_predict"] for df in dfs_opt_test])
Y_true = dfs_opt_test[0]["E_barrier_opt"].to_numpy().astype(float)

# Sanity check that all have the same reference energies
assert np.all(
    [
        np.allclose(Y_true, dfs_opt_test[i]["E_barrier_opt"].to_numpy().astype(float))
        for i in range(len(dfs_opt_test))
    ]
)

# Selection criteria
select_traj = origin == "traj"
select_3A = d <= 3
select_2A = d <= 2

mae_all = np.array(
    [mean_absolute_error(Y_true, Y_predict_per_seed[s]) for s in painn_seeds]
)
mae_traj = np.array(
    [
        mean_absolute_error(Y_true[select_traj], Y_predict_per_seed[s][select_traj])
        for s in painn_seeds
    ]
)
mae_traj3A = np.array(
    [
        mean_absolute_error(
            Y_true[select_traj * select_3A],
            Y_predict_per_seed[s][select_traj * select_3A],
        )
        for s in painn_seeds
    ]
)
mae_traj2A = np.array(
    [
        mean_absolute_error(
            Y_true[select_traj * select_2A],
            Y_predict_per_seed[s][select_traj * select_2A],
        )
        for s in painn_seeds
    ]
)

print(
    "MAE ALL:",
    [f"{x:.2f}" for x in mae_all],
    f"{mae_all.mean():.2f}",
    f"{mae_all.std():.2f}",
)
print(
    "MAE TRAJ:",
    [f"{x:.2f}" for x in mae_traj],
    f"{mae_traj.mean():.2f}",
    f"{mae_traj.std():.2f}",
)
print(
    "MAE 3A:",
    [f"{x:.2f}" for x in mae_traj3A],
    f"{mae_traj3A.mean():.2f}",
    f"{mae_traj3A.std():.2f}",
)
print(
    "MAE 2A:",
    [f"{x:.2f}" for x in mae_traj2A],
    f"{mae_traj2A.mean():.2f}",
    f"{mae_traj2A.std():.2f}",
)

# Ensemble
print("Ensemble")
Y_predict_ensemble = np.mean(Y_predict_per_seed, axis=0)

mae_all_ensemble = mean_absolute_error(Y_true, Y_predict_ensemble)
mae_traj_ensemble = mean_absolute_error(
    Y_true[select_traj], Y_predict_ensemble[select_traj]
)
mae_traj3A_ensemble = mean_absolute_error(
    Y_true[select_traj * select_3A], Y_predict_ensemble[select_traj * select_3A]
)
mae_traj2A_ensemble = mean_absolute_error(
    Y_true[select_traj * select_2A], Y_predict_ensemble[select_traj * select_2A]
)

mae_all_std = np.std(
    [
        mean_absolute_error(Y_true, Y_predict_per_seed[seed_i])
        for seed_i in range(Y_predict_per_seed.shape[0])
    ]
)
mae_traj_std = np.std(
    [
        mean_absolute_error(
            Y_true[select_traj], Y_predict_per_seed[seed_i][select_traj]
        )
        for seed_i in range(Y_predict_per_seed.shape[0])
    ],
)
mae_traj3A_std = np.std(
    [
        mean_absolute_error(
            Y_true[select_traj * select_3A],
            Y_predict_per_seed[seed_i][select_traj * select_3A],
        )
        for seed_i in range(Y_predict_per_seed.shape[0])
    ],
)
mae_traj2A_std = np.std(
    [
        mean_absolute_error(
            Y_true[select_traj * select_2A],
            Y_predict_per_seed[seed_i][select_traj * select_2A],
        )
        for seed_i in range(Y_predict_per_seed.shape[0])
    ],
)

print("MAE ALL:", f"{mae_all_ensemble:.2f}")
print("MAE TRAJ:", f"{mae_traj_ensemble:.2f}")
print("MAE 3A:", f"{mae_traj3A_ensemble:.2f}")
print("MAE 2A:", f"{mae_traj2A_ensemble:.2f}")

PaiNN_paper_mae_3A_traj = 4.93
PaiNN_paper_mae_2A_traj = 3.64

print(
    f"ALL Traj | PaiNN+GPR: {(res := mae_traj_ensemble):.2f} ± {(un := mae_traj_std):.2f}, "
)
print(
    f"3A | PaiNN+GPR: {(res := mae_traj3A_ensemble):.2f} ± {(un := mae_traj3A_std):.2f}, "
    f"Pure PaiNN (other paper): {(ref := PaiNN_paper_mae_3A_traj):.2f}, "
    f"Improvement: {1 - res / ref:.1%} ± {un / ref:.1%}"
)
print(
    f"2A | PaiNN+GPR: {(res := mae_traj2A_ensemble):.2f} ± {(un := mae_traj2A_std):.2f}, "
    f"Pure PaiNN (other paper): {(ref := PaiNN_paper_mae_2A_traj):.2f}, "
    f"Improvement: {1 - res / ref:.1%} ± {un / ref:.1%}"
)

# Plot mae vs cut-off distance
ds = np.sort(np.unique(d[select_traj]))
maes_per_seed = np.array(
    [
        [
            mean_absolute_error(
                Y_true[select_traj * (d <= d_max)],
                Y_predict_per_seed[s][select_traj * (d <= d_max)],
            )
            for d_max in ds
        ]
        for s in painn_seeds
    ]
)
maes_mean = np.mean(maes_per_seed, axis=0)
maes_std = np.std(maes_per_seed, axis=0, ddof=1)

color_seeds = "#485696"
color_mean = "#FF5722"
plot_properties = {
    "font.size": 35,
    "xtick.major.size": 15,
    "ytick.major.size": 15,
    "xtick.major.width": 2,
    "ytick.major.width": 2,
    "xtick.minor.size": 15,
    "ytick.minor.size": 15,
    "xtick.minor.width": 2,
    "ytick.minor.width": 2,
    "axes.linewidth": 2,
    "legend.fontsize": 26,
}
mpl.rcParams.update(plot_properties)
fig, axes = plt.subplots(1, 1, figsize=(14, 10), sharex=True)
axes = [axes]
axes[0].fill_between(
    ds,
    maes_mean - maes_std,
    maes_mean + maes_std,
    alpha=0.2,
    color=color_mean,
    label="Standard\nDeviation",
    zorder=2,
)
for i_s, s in enumerate(painn_seeds):
    label = "Run" if i_s == 0 else None
    axes[0].plot(
        ds,
        maes_per_seed[s],
        "s",
        color=color_seeds,
        markersize=10,
        markeredgecolor="black",
        label=label,
        zorder=1,
    )
axes[0].plot(ds, maes_mean, "-", color=color_mean, lw=4, label="Mean", zorder=3)
axes[0].set_xlim(ds.min() - 0.1, ds.max() + 0.1)
axes[0].grid(True, alpha=0.3, which="both", ls="-")
axes[0].set_ylabel(r"MAE [kcal/mol]")

axes[0].set_xticks(np.arange(1.0, 3.01, 0.25))
axes[0].set_xticklabels(axes[0].get_xticks(), rotation=-45)
axes[0].set_yticks(np.arange(1.5, 7.01, 0.5))

axes[0].set_xlabel(r"Maximum distance [Å]")
plt.legend(loc="upper right", ncol=1, borderaxespad=0, frameon=False)
plt.tight_layout()
fig.savefig(
    cwd / "figures" / "two_stage_learning.png",
    pad_inches=0.1,
    dpi=400,
    bbox_inches="tight",
)
plt.show()


# Plot optimized predictions
fig, ax = plt.subplots(figsize=(10, 10))

ax.grid(True, alpha=0.2)
selections = [select_traj, select_traj * select_2A]
selection_name = [r"$d \geq 2\AA$", r"$d \leq 2\AA$"]
for i, selection in enumerate(selections):
    ax.scatter(
        Y_true[selection],
        Y_predict_per_seed[0][selection],
        s=30,
        marker="o",
        alpha=1.0,
        linewidths=0.5,
        edgecolors="black",
        label=rf"{selection_name[i]}",
        zorder=3,
    )

ax.set_aspect("equal", "box")
ax.set_xlabel(r"$\Delta E^{True}$ [kcal/mol]")
ax.set_ylabel(r"$\Delta E^{Predicted}$ [kcal/mol]")

ax_min = 5
ax_max = np.max([Y_true[select_traj], Y_predict_per_seed[0][select_traj]])

ax.set_xlim(ax_min, ax_max)
ax.set_ylim(ax_min, ax_max)

# Check that no point is outside limits
assert (
    ax_min >= ax.get_xlim()[0]
    and ax_max <= ax.get_xlim()[1]
    and ax_min >= ax.get_ylim()[0]
    and ax_max <= ax.get_ylim()[1]
)

ticks = np.arange(ax.set_xlim()[0], ax.set_xlim()[1] + 10, 10)
ax.set_xticks(ticks)
ax.set_xticklabels(ticks.astype(int), rotation=-45)
ax.set_yticks(ticks)
ax.set_yticklabels(ticks.astype(int), rotation=0)
diag = np.linspace(np.max([ax.get_xlim()[0], 0.0]), ax.get_xlim()[1], 1000)
ax.plot(diag, diag, color="black", ls="--", zorder=0, lw=2)
ax.legend(loc="upper left", ncol=1, borderaxespad=0, frameon=False, markerscale=3)
ax.set_title("Trajectory Optimized Barriers", fontsize=32)

plt.tight_layout()
plt.savefig(
    cwd / "figures" / "two_stage_prediction_scatter.png", dpi=400, bbox_inches="tight"
)
plt.show()
