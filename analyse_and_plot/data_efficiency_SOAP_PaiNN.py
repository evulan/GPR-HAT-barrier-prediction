"""Plots the number of training points vs the MAE of the GPR SOAP and PaiNN methods"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit
from matplotlib.ticker import StrMethodFormatter, NullFormatter
from pathlib import Path

cwd = Path(__file__).resolve().parent

# Load the postprocessed statistics and save to different variables
# Notation: GPR both -> GPR SOAP trained on traj and synth, GPR traj -> GPR SOAP trained on traj only
df_stats = pd.read_pickle(cwd / "results" / "mae_scenarios.pkl")
n_train = df_stats["n_train"].to_numpy()
fracs = df_stats["frac"].to_numpy()

painn_individual_mean_2A = df_stats["PaiNN Individual MEAN 2A traj"].to_numpy()
painn_individual_std_2A = df_stats["PaiNN Individual STD 2A traj"].to_numpy()

painn_ensemble_mean_2A = df_stats["PaiNN Ensemble MEAN 2A traj"].to_numpy()
painn_ensemble_std_2A = df_stats["PaiNN Ensemble STD 2A traj"].to_numpy()

painn_individual_mean_3A = df_stats["PaiNN Individual MEAN 3A traj"].to_numpy()
painn_individual_std_3A = df_stats["PaiNN Individual STD 3A traj"].to_numpy()

painn_ensemble_mean_3A = df_stats["PaiNN Ensemble MEAN 3A traj"].to_numpy()
painn_ensemble_std_3A = df_stats["PaiNN Ensemble STD 3A traj"].to_numpy()

painn_individual_mean_all_traj = df_stats["PaiNN Individual MEAN all traj"].to_numpy()
painn_individual_std_all_traj = df_stats["PaiNN Individual STD all traj"].to_numpy()

painn_ensemble_mean_all_traj = df_stats["PaiNN Ensemble MEAN all traj"].to_numpy()
painn_ensemble_std_all_traj = df_stats["PaiNN Ensemble STD all traj"].to_numpy()


GPR_both_mean_2A = df_stats["GPR Both MEAN 2A traj"].to_numpy()
GPR_both_std_2A = df_stats["GPR Both STD 2A traj"].to_numpy()

GPR_both_mean_3A = df_stats["GPR Both MEAN 3A traj"].to_numpy()
GPR_both_std_3A = df_stats["GPR Both STD 3A traj"].to_numpy()


GPR_traj_mean_2A = df_stats["GPR Traj MEAN 2A traj"].to_numpy()
GPR_traj_std_2A = df_stats["GPR Traj STD 2A traj"].to_numpy()

GPR_traj_mean_3A = df_stats["GPR Traj MEAN 3A traj"].to_numpy()
GPR_traj_std_3A = df_stats["GPR Traj STD 3A traj"].to_numpy()


GPR_both_mean_all_traj = df_stats["GPR Both MEAN all traj"].to_numpy()
GPR_both_std_all_traj = df_stats["GPR Both STD all traj"].to_numpy()

GPR_traj_mean_all_traj = df_stats["GPR Traj MEAN all traj"].to_numpy()
GPR_traj_std_all_traj = df_stats["GPR Traj STD all traj"].to_numpy()


def f2n(f, n_max):
    """Training fraction to number of training points"""
    return (f * n_max / 2).astype(int) * 2


def n2f(n, n_max):
    """Number of training points to fraction of training points"""
    return np.ceil(n / n_max * 10000) / 10000


# Sanity check
assert np.allclose(
    f2n(fracs, n_train.max()), n_train
), f"\n{fracs.tolist()}\n{f2n(fracs, n_train.max()).tolist()}\n{n_train.tolist()}"
assert np.allclose(
    n2f(n_train, n_train.max()), fracs, atol=1e-4
), f"\n{n_train.tolist()}\n{n2f(n_train, n_train.max())}\n{fracs.tolist()}"


def power(x, k, a):
    """Power law function to fit"""
    return a * (x**k)


def estimate_n_threshold(a, k, target_MAE=2):
    """Estimates number of training points needed to reach target MAE, based on power law parameters"""
    n_threshold = np.floor(np.exp(np.log(target_MAE / a) / k)).astype(
        int
    )  # in general will be slightly above target MAE
    assert np.isclose(power(n_threshold, k, a), target_MAE, atol=0.01)
    while (
        a * (n_threshold**k) > target_MAE
    ):  # add training points until at or just below target
        n_threshold += 1
    assert np.isclose(power(n_threshold, k, a), target_MAE, atol=0.0001)
    return n_threshold


def interception(a1, k1, a2, k2):
    """Compute the number of training points n at intercept of a1*(n**k1) and a2*(n**k2)"""
    return (a1 / a2) ** (1 / (k2 - k1))


def get_loglog_fit(fs, maes, sigma, n_max, lim=None, name=""):
    """Fit a power law to MAEs"""
    mask = ~np.isnan(maes)
    fs = fs[mask]
    ns = f2n(fs, n_max)
    maes = maes[mask]
    sigma = sigma[mask]

    p0 = [-0.3, 30]
    popt, pcov = curve_fit(power, ns, maes, p0=p0, sigma=sigma, maxfev=10000)
    perr = np.sqrt(np.diag(pcov))

    f_plot = np.linspace(*lim, 1000)
    n_plot = f2n(f_plot, n_max)
    mae_plot = power(n_plot, *popt)

    # Power law parameters
    k = popt[0]
    err_k = perr[0]
    a = popt[1]
    err_a = perr[1]

    n_threshold = estimate_n_threshold(a, k, target_MAE=2)
    factor_threshold = n_threshold / n_max
    print(
        f"Best fit: a={a:.1f} \u00b1 {err_a:.1f}, k={k:.3f} \u00b1 {err_k:.3f}, "
        f"target n: {np.round(n_threshold, -3)} (factor: {factor_threshold:.1f}) for {name}"
    )
    return (a, k), f_plot, mae_plot


# Fit power law to all combinations of methods and 2A and 3A cut-offs
params__gpr_both_3A, fracs_fit, mae_fit_plot__gpr_both_3A = get_loglog_fit(
    fracs,
    GPR_both_mean_3A,
    sigma=GPR_both_std_3A,
    n_max=n_train.max(),
    lim=[0.01, 1.0],
    name="GPR Traj & Synth 3A",
)
params__gpr_both_2A, _, mae_fit_plot__gpr_both_2A = get_loglog_fit(
    fracs,
    GPR_both_mean_2A,
    sigma=GPR_both_std_2A,
    n_max=n_train.max(),
    lim=[0.01, 1.0],
    name="GPR Traj & Synth 2A",
)
params__gpr_traj_3A, _, mae_fit_plot__gpr_traj_3A = get_loglog_fit(
    fracs,
    GPR_traj_mean_3A,
    sigma=GPR_traj_std_3A,
    n_max=n_train.max(),
    lim=[0.01, 1.0],
    name="GPR Traj 3A",
)
params__gpr_traj_2A, _, mae_fit_plot__gpr_traj_2A = get_loglog_fit(
    fracs,
    GPR_traj_mean_2A,
    sigma=GPR_traj_std_2A,
    n_max=n_train.max(),
    lim=[0.01, 1.0],
    name="GPR Traj 2A",
)

params__painn_individual_both_3A, _, mae_fit_plot__painn_individual_both_3A = (
    get_loglog_fit(
        fracs,
        painn_individual_mean_3A,
        sigma=painn_individual_std_3A,
        n_max=n_train.max(),
        lim=[0.01, 1.0],
        name="PaiNN Individual 3A",
    )
)
params__painn_individual_both_2A, _, mae_fit_plot__painn_individual_both_2A = (
    get_loglog_fit(
        fracs,
        painn_individual_mean_2A,
        sigma=painn_individual_std_2A,
        n_max=n_train.max(),
        lim=[0.01, 1.0],
        name="PaiNN Individual 2A",
    )
)
params__painn_ensemble_traj_3A, _, mae_fit_plot__painn_ensemble_traj_3A = (
    get_loglog_fit(
        fracs,
        painn_ensemble_mean_3A,
        sigma=painn_ensemble_std_3A,
        n_max=n_train.max(),
        lim=[0.01, 1.0],
        name="PaiNN Ensemble 3A",
    )
)
params__painn_ensemble_traj_2A, _, mae_fit_plot__painn_ensemble_traj_2A = (
    get_loglog_fit(
        fracs,
        painn_ensemble_mean_2A,
        sigma=painn_ensemble_std_2A,
        n_max=n_train.max(),
        lim=[0.01, 1.0],
        name="PaiNN Ensemble 2A",
    )
)

params__gpr_both_all_traj, _, mae_fit_plot__gpr_both_all_traj = get_loglog_fit(
    fracs,
    GPR_both_mean_all_traj,
    sigma=GPR_both_std_all_traj,
    n_max=n_train.max(),
    lim=[0.01, 1.0],
    name="GPR Traj & Synth ALL TRAJ",
)
params__gpr_traj_all_traj, _, mae_fit_plot__gpr_traj_all_traj = get_loglog_fit(
    fracs,
    GPR_traj_mean_all_traj,
    sigma=GPR_traj_std_all_traj,
    n_max=n_train.max(),
    lim=[0.01, 1.0],
    name="GPR Traj ALL TRAJ",
)
(
    params__painn_individual_both_all_traj,
    _,
    mae_fit_plot__painn_individual_both_all_traj,
) = get_loglog_fit(
    fracs,
    painn_individual_mean_all_traj,
    sigma=painn_individual_std_all_traj,
    n_max=n_train.max(),
    lim=[0.01, 1.0],
    name="PaiNN Individual ALL TRAJ",
)
params__painn_ensemble_traj_all_traj, _, mae_fit_plot__painn_ensemble_traj_all_traj = (
    get_loglog_fit(
        fracs,
        painn_ensemble_mean_all_traj,
        sigma=painn_ensemble_std_all_traj,
        n_max=n_train.max(),
        lim=[0.01, 1.0],
        name="PaiNN Ensemble ALL TRAJ",
    )
)

# Print intersections
print(
    f"ALL Traj: Interception of GPR_both and PaiNN ensemble at n ~ {np.round(interception(*params__gpr_both_all_traj, *params__painn_ensemble_traj_all_traj), -3):.0f}"
)
print(
    f"ALL 3A Traj: Interception of GPR_both and PaiNN ensemble at n ~ {np.round(interception(*params__gpr_both_3A, *params__painn_ensemble_traj_3A), -3):.0f}"
)
print(
    f"ALL 2A Traj: Interception of GPR_both and PaiNN ensemble at n ~ {np.round(interception(*params__gpr_both_2A, *params__painn_ensemble_traj_2A), -3):.0f}"
)

# Plot figures
plot_properties = {
    "font.size": 26,
    "xtick.major.size": 15,
    "ytick.major.size": 15,
    "xtick.major.width": 2,
    "ytick.major.width": 2,
    "xtick.minor.size": 15,
    "ytick.minor.size": 15,
    "xtick.minor.width": 2,
    "ytick.minor.width": 2,
    "axes.linewidth": 2,
    "legend.fontsize": 22,
}
mpl.rcParams.update(plot_properties)
fig, axes = plt.subplots(1, 2, figsize=(24, 10), sharey=True)

# PaiNN  3A
axes[0].errorbar(
    fracs,
    painn_individual_mean_3A,
    yerr=painn_individual_std_3A,
    fmt="s",
    linestyle="",
    lw=2,
    markersize=10,
    label=r"$\mathregular{PaiNN_{Ind}}$",
    color="#fec601",
    alpha=1,
    capsize=15,
    elinewidth=3,
    markeredgewidth=3,
    zorder=1,
)
axes[0].plot(
    fracs_fit,
    mae_fit_plot__painn_individual_both_3A,
    "--",
    markersize=10,
    color="#fec601",
    alpha=1,
    lw=2,
    zorder=0,
)

axes[0].errorbar(
    fracs,
    painn_ensemble_mean_3A,
    yerr=painn_ensemble_std_3A,
    fmt="s",
    linestyle="",
    lw=2,
    markersize=10,
    label=r"$\mathregular{PaiNN_{Ens}}$",
    color="#ea7317",
    alpha=1,
    capsize=15,
    elinewidth=3,
    markeredgewidth=3,
    zorder=3,
)
axes[0].plot(
    fracs_fit,
    mae_fit_plot__painn_ensemble_traj_3A,
    "--",
    markersize=10,
    color="#ea7317",
    alpha=1,
    lw=2,
    zorder=2,
)

# PaiNN  2A
axes[1].errorbar(
    fracs,
    painn_individual_mean_2A,
    yerr=painn_individual_std_2A,
    fmt="s",
    linestyle="",
    lw=2,
    markersize=10,
    label=r"$\mathregular{PaiNN_{Ind}}$",
    color="#fec601",
    alpha=1,
    capsize=15,
    elinewidth=3,
    markeredgewidth=3,
    zorder=1,
)
axes[1].plot(
    fracs_fit,
    mae_fit_plot__painn_individual_both_2A,
    "--",
    markersize=10,
    color="#fec601",
    alpha=1,
    lw=2,
    zorder=0,
)

axes[1].errorbar(
    fracs,
    painn_ensemble_mean_2A,
    yerr=painn_ensemble_std_2A,
    fmt="s",
    linestyle="",
    lw=2,
    markersize=10,
    label=r"$\mathregular{PaiNN_{Ens}}$",
    color="#ea7317",
    alpha=1,
    capsize=15,
    elinewidth=3,
    markeredgewidth=3,
    zorder=3,
)
axes[1].plot(
    fracs_fit,
    mae_fit_plot__painn_ensemble_traj_2A,
    "--",
    markersize=10,
    color="#ea7317",
    alpha=1,
    lw=2,
    zorder=2,
)


# GPR both 3A
axes[0].errorbar(
    fracs,
    GPR_both_mean_3A,
    yerr=GPR_both_std_3A,
    fmt="o",
    linestyle="",
    markersize=10,
    lw=2,
    label=r"$\mathregular{SOAP_{Full}}$",
    color="#06aed5",
    capsize=15,
    elinewidth=3,
    markeredgewidth=3,
    zorder=5,
)
axes[0].plot(
    fracs_fit,
    mae_fit_plot__gpr_both_3A,
    "--",
    markersize=10,
    color="#06aed5",
    alpha=1,
    lw=2,
    label="power-law fit",
    zorder=4,
)

# GPR both 2A
axes[1].errorbar(
    fracs,
    GPR_both_mean_2A,
    yerr=GPR_both_std_2A,
    fmt="o",
    linestyle="",
    markersize=10,
    lw=2,
    label=r"$\mathregular{SOAP_{Full}}$",
    color="#06aed5",
    capsize=15,
    elinewidth=3,
    markeredgewidth=3,
    zorder=5,
)
axes[1].plot(
    fracs_fit,
    mae_fit_plot__gpr_both_2A,
    "--",
    markersize=10,
    color="#06aed5",
    alpha=1,
    lw=2,
    label="power-law fit",
    zorder=4,
)


# GPR traj 3A
axes[0].errorbar(
    fracs,
    GPR_traj_mean_3A,
    yerr=GPR_traj_std_3A,
    fmt="o",
    linestyle="",
    markersize=10,
    lw=2,
    label=r"$\mathregular{SOAP_{Traj}}$",
    color="#086788",
    capsize=15,
    elinewidth=3,
    markeredgewidth=3,
    zorder=7,
)
axes[0].plot(
    fracs_fit,
    mae_fit_plot__gpr_traj_3A,
    "--",
    markersize=10,
    color="#086788",
    alpha=1,
    lw=2,
    zorder=6,
)

# GPR traj 2A
axes[1].errorbar(
    fracs,
    GPR_traj_mean_2A,
    yerr=GPR_traj_std_2A,
    fmt="o",
    linestyle="",
    markersize=10,
    lw=2,
    label=r"$\mathregular{SOAP_{Traj}}$",
    color="#086788",
    capsize=15,
    elinewidth=3,
    markeredgewidth=3,
    zorder=7,
)
axes[1].plot(
    fracs_fit,
    mae_fit_plot__gpr_traj_2A,
    "--",
    markersize=10,
    color="#086788",
    alpha=1,
    lw=2,
    zorder=6,
)


def frac2absolute(f):
    return f2n(f, n_train.max())


def absolute2frac(n):
    return n2f(n, n_train.max())


axes_top_0 = axes[0].secondary_xaxis("top", functions=(frac2absolute, absolute2frac))
axes_top_0.set_xlabel(r"Training points", fontsize=26, loc="left")
axes_top_0.set_xticks(n_train)
axes_top_0.set_xticklabels(
    [f"{t}" for t in axes_top_0.get_xticks()], rotation=0, fontsize=26
)

axes_top_1 = axes[1].secondary_xaxis("top", functions=(frac2absolute, absolute2frac))
axes_top_1.set_xlabel(r"Training points", fontsize=26, loc="left")
axes_top_1.set_xticks(n_train)
axes_top_1.set_xticklabels(
    [f"{t}" for t in axes_top_0.get_xticks()], rotation=0, fontsize=26
)

for ax in axes.flatten():
    ax.grid(True, alpha=0.3, which="both", ls="-")
    ax.set_xlabel(r"of all training data (Trajectory & Synthetic)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.0f}"))
    ax.yaxis.set_minor_formatter(StrMethodFormatter("{x:.0f}"))
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:.0%}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    legend_handles, _ = ax.get_legend_handles_labels()
    legend_handles[0] = Line2D(
        [0], [0], color="black", linewidth=2, linestyle="--", label="Power-law fit"
    )
    ax.legend(loc="upper right", handles=legend_handles)

axes[0].set_title(r"Test: $d\leq 3\AA$ Trajectory", pad=20)
axes[1].set_title(r"Test: $d\leq 2\AA$ Trajectory", pad=20)
axes[0].set_ylabel(r"MAE$\:\left[\frac{kcal}{mol}\right]$")

plt.tight_layout()
plt.savefig(
    "figures/compare_with_ensemble_paper_loglog.png", dpi=400, bbox_inches="tight"
)
plt.show()
