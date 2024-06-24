"""Script which imports the SOAP and PaiNN results and creates statistics files for further analysis."""

import numpy as np
import pandas as pd
from pathlib import Path


def MAE(*, errs):
    """Convenience function to get MAE from errors: err = true - pred"""
    return np.mean(np.abs(errs))


cwd = Path(__file__).resolve().parent
project_root = cwd.parent

# Load data and results from SOAP GPR and PaiNN

# Load whole dataset
path_atoms = project_root / "data" / "atoms.pkl"
df_atoms = pd.read_pickle(path_atoms)
df_atoms = df_atoms[df_atoms.split == "test"]

# Load SOAP GPR prediction results for multiple training subset fractions and seed when trained on both traj & synth
path_gpr_train_both = (
    project_root / "SOAP_GPR" / "results" / "GPR_predictions_train_both.npz"
)
print(path_gpr_train_both.resolve())
x_GPR_both = np.load(path_gpr_train_both, allow_pickle=True)

# Load SOAP GPR prediction results for multiple training subset fractions and seed when trained on both traj only
path_gpr_train_traj = (
    project_root / "SOAP_GPR" / "results" / "GPR_predictions_train_traj.npz"
)
x_GPR_traj = np.load(path_gpr_train_traj, allow_pickle=True)

# Collect the PaiNN prediction results
# Results folder:
path_painn_results_dir = (
    project_root / "PaiNN" / "usage" / "results" / "data_efficiency"
)
# Paths to result files of every rank
paths_painn_train_both = np.sort(
    [f for f in path_painn_results_dir.glob(f"rank_*_test.npz")]
)
# Get number of ranks used
painn_ranks = np.array(
    [
        int(res_file.stem.split("_")[-2])
        for res_file in np.sort(
            [f for f in path_painn_results_dir.glob(f"rank_[0-9]_test.npz")]
        )
    ]
)

print(
    f"GPR train both: {path_gpr_train_both}\nGPR train traj: {path_gpr_train_traj}\nPaiNN ranks both {painn_ranks} "
    f"at\n{[f.name for f in paths_painn_train_both]}"
)


def res_array_to_dict(res_array, log, prefix):
    """Creates a dataframe with the prediction results with all runs of either PaiNN or GPR"""
    # For PaiNN case
    if prefix == "PaiNN":
        columns = [
            "transition_hash",
            "path",
            "direction",
            "E_barrier",
            "(first errors)",
        ]
        # Each column will be named as E_PaiNN={run_hash},
        # where run_hash is a unique hash for each run
        # In these columns the predicted energy barrier of a specific run will be saved
        prediction_columns = [
            "E_" + prefix + "=" + h.split("<<<")[1].split(">>>")[0]
            for h in log["train_all"]["hashid"]
        ]
        df = pd.DataFrame.from_records(
            res_array, columns=columns + prediction_columns
        ).drop(columns=["(first errors)"])

        # Create a unique index for each transition and direction
        df["hash_direction"] = [
            f'{df["transition_hash"].iloc[i]}_{df["direction"].iloc[i]}'
            for i in range(df.shape[0])
        ]
        df = df.set_index(["hash_direction"])
    # For SOAP GPR cases
    elif prefix in ["GPR_both", "GPR_traj"]:
        # Metadata columns
        columns = [
            "opt",
            "E_forward_opt",
            "E_backward_opt",
            "d",
            "origin",
            "split1",
            "n_atoms",
            "meta_hash",
            "metafile_id",
            "transition_hash",
            "split2",
            "E_barrier",
            "direction",
        ]

        # List of unique prediction run hashes
        run_hashes = list(log.item().keys())
        for s in ["hashid", "frac", "seed", "final_params"]:
            run_hashes.remove(s)  # remove not hashids

        # Each column will be named as E_{method}={run_hash},
        # where run_hash is a unique hash for each run and method is either GPR_both or GPR_traj
        # In these columns the predicted energy barrier of a specific run will be saved
        prediction_columns = (
            np.array(
                [
                    ["E_" + prefix + "=" + h, "Sigma_" + prefix + "=" + h]
                    for h in run_hashes
                ]
            )
            .flatten()
            .tolist()
        )
        df = pd.DataFrame.from_records(res_array, columns=columns + prediction_columns)

        # Sanity check
        assert np.array_equal(df.split1.to_numpy(), df.split2.to_numpy())

        # Create a unique index for each transition and direction
        df["hash_direction"] = [
            f'{df["transition_hash"].iloc[i]}_{df["direction"].iloc[i]}'
            for i in range(df.shape[0])
        ]
        df = df.set_index(["hash_direction"])
    else:
        raise NotImplementedError()
    return df


def lookup_table_PaiNN(paths_painn_train_both):
    """Creates a lookup table indexed by the run hash with information about data fraction, seed and ensemble rank"""
    dfs = []
    # For each rank load the predictions. At the end concatenate the results of all ranks
    for path_painn_train_both in paths_painn_train_both:
        x = np.load(path_painn_train_both, allow_pickle=True)
        log = x["log"].item()  # meta data stored here
        df = pd.DataFrame.from_dict(log["train_all"]).rename(
            columns={"hashid": "run_hash"}
        )

        # untestd
        if "GNN_E_barrier_predicted" in df.columns:
            df.rename(
                columns={
                    "GNN_E_barrier_predicted": f"E_barrier_predict-{log['hashid'][0]}"
                },
                inplace=True,
            )

        # Only require actual run hash
        df.loc[:, "run_hash"] = df["run_hash"].apply(
            lambda x: x.split("<<<")[1].split(">>>")[0]
        )
        df.set_index("run_hash", inplace=True)
        dfs.append(df)
    df = pd.concat(dfs, axis=0)
    df = df.rename(columns={"data_frac": "frac"}).astype(
        {"frac": "float64", "sample_seed": "int64", "rank": "int64"}
    )
    # Sort dataframe by fraction, then by ensemble rank and finally by the seed
    df.sort_values(
        ["frac", "rank", "sample_seed"], ascending=[True, True, True], inplace=True
    )
    return df


def lookup_table_GPR(paths_gpr):
    """Creates a lookup table indexed by the run hash with information about data fraction, seed and optimized param."""
    # Load metadata
    x = np.load(paths_gpr, allow_pickle=True)
    log = x["log"].item()
    run_hashes = list(log.keys())
    for s in ["hashid", "frac", "seed", "final_params"]:
        run_hashes.remove(s)  # filter to only include run hashes
    data = [
        [
            run_hash,
            log[run_hash]["n_train"],
            log[run_hash]["seed"],
            log[run_hash]["final_params"],
        ]
        for run_hash in run_hashes
    ]
    columns = ["run_hash", "frac", "seed", "final_params"]
    # Create metadata look up table
    df = pd.DataFrame.from_records(data, columns=columns)
    df = df.astype(
        {"run_hash": "string", "frac": "float64", "seed": "int64"}
    ).set_index("run_hash")
    df.sort_values(
        ["frac", "seed"], ascending=[True, True], inplace=True
    )  # Sort by data fraction, then seed
    return df


def df_pred_PaiNN(paths_painn_train_both):
    """Create table with all PaiNN predictions."""
    df_pred = None
    # Iteratively concatenate PaiNN predictions for each ensemble rank
    for path_painn_train_both in paths_painn_train_both:
        print(f"Loading {path_painn_train_both}")
        # Load predictions and metadata
        x = np.load(path_painn_train_both, allow_pickle=True)
        df_painn_rank = res_array_to_dict(x["df_pred"], x["log"].item(), "PaiNN")

        if df_pred is None:  # if first iteration
            df_pred = df_painn_rank.copy()
        else:  # each subsequent iteration is concatenated
            # sanity checks
            assert (
                len(
                    np.setdiff1d(
                        df_painn_rank.index.to_numpy(), df_pred.index.to_numpy()
                    )
                )
                == 0
            )
            assert np.array_equal(
                df_painn_rank.index.to_numpy(), df_pred.index.to_numpy()
            )
            assert np.all(
                np.isin(df_painn_rank.index.to_numpy(), df_pred.index.to_numpy())
            )

            # filter out columns which are not predictions
            filter_predictions = [c for c in df_painn_rank if c.startswith("E_PaiNN=")]
            df_pred = pd.concat(
                [df_pred, df_painn_rank.loc[:, filter_predictions].copy()], axis=1
            )
            assert df_pred.shape[0] == df_painn_rank.shape[0]
    return df_pred


def merge_PaiNN_GPR(paths_painn_train_both, x_GPR_both, x_GPR_traj):
    """Create a dataframe with all predictions, i.e. GPR & PaiNN results"""

    # Get dataframe with PaiNN results
    df_PaiNN = df_pred_PaiNN(paths_painn_train_both)

    # Get dataframes of SOAP GPR results (training: trajectory+synthetic & only trajectory)
    df_GPR_both = res_array_to_dict(
        x_GPR_both["df_pred"], x_GPR_both["log"], "GPR_both"
    )
    df_GPR_traj = res_array_to_dict(
        x_GPR_traj["df_pred"], x_GPR_traj["log"], "GPR_traj"
    )

    # sanity check: all dataframes have predictions for the same transition & directions
    assert df_PaiNN.shape[0] == df_GPR_both.shape[0] == df_GPR_traj.shape[0]
    assert set(df_PaiNN.index.to_numpy()) == set(df_GPR_both.index.to_numpy())
    assert set(df_PaiNN.index.to_numpy()) == set(df_GPR_traj.index.to_numpy())

    # Sort GPR predictions according to PaiNN index
    df_GPR_both = df_GPR_both.loc[df_PaiNN.index]
    df_GPR_traj = df_GPR_traj.loc[df_PaiNN.index]

    # sanity check that all in correct order
    assert np.array_equal(df_PaiNN.index.to_numpy(), df_GPR_both.index.to_numpy())
    assert np.array_equal(df_PaiNN.index.to_numpy(), df_GPR_traj.index.to_numpy())
    assert np.allclose(df_PaiNN.E_barrier.to_numpy(), df_GPR_both.E_barrier.to_numpy())
    assert np.allclose(df_PaiNN.E_barrier.to_numpy(), df_GPR_traj.E_barrier.to_numpy())

    # filter out only the predictions
    df_Painn_predictions = df_PaiNN.loc[
        :, [c for c in df_PaiNN.columns if c.startswith("E_PaiNN=")]
    ].copy()
    df_GPR_both_predictions = df_GPR_both.loc[
        :,
        [
            c
            for c in df_GPR_both.columns
            if c.startswith("E_GPR_both=") or c.startswith("Sigma_GPR_both=")
        ],
    ].copy()
    df_GPR_traj_predictions = df_GPR_traj.loc[
        :,
        [
            c
            for c in df_GPR_traj.columns
            if c.startswith("E_GPR_traj=") or c.startswith("Sigma_GPR_traj=")
        ],
    ].copy()

    # create one large dataframe with all predictions
    df = pd.concat(
        [
            df_PaiNN.loc[
                :, ["transition_hash", "direction", "path", "E_barrier"]
            ].copy(),
            df_GPR_both.loc[:, ["origin", "d"]].copy(),
            df_Painn_predictions,
            df_GPR_both_predictions,
            df_GPR_traj_predictions,
        ],
        axis=1,
    )

    return df


# get one dataframe which has all predictions
print("Load and Merge")
df = merge_PaiNN_GPR(paths_painn_train_both, x_GPR_both, x_GPR_traj)
print(df)

# Create look up tables that give the training set fraction used and seed
print("Create lookup")
lookup_PaiNN = lookup_table_PaiNN(paths_painn_train_both)
lookup_GPR_both = lookup_table_GPR(path_gpr_train_both)
lookup_GPR_traj = lookup_table_GPR(path_gpr_train_traj)
print("PaiNN\n", lookup_PaiNN, lookup_PaiNN.columns, lookup_PaiNN.dtypes)
print("GPR Both\n", lookup_GPR_both, lookup_GPR_both.columns, lookup_GPR_both.dtypes)
print("GPR Traj\n", lookup_GPR_traj, lookup_GPR_traj.columns, lookup_GPR_traj.dtypes)

# collect the training set fractions used during training (traj.-only training used only ~half of data)
fracs = {
    "PaiNN": np.unique(lookup_PaiNN["frac"]),
    "GPR_both": np.unique(lookup_GPR_both["frac"]),
    "GPR_traj": np.unique(lookup_GPR_traj["frac"]),
}

fracs["all"] = np.unique(
    fracs["PaiNN"].tolist() + fracs["GPR_both"].tolist() + fracs["GPR_traj"].tolist()
)
print(fracs)

# collect all seeds used to shuffle training data to create fractional training subsets
seeds = {
    "PaiNN": np.unique(lookup_PaiNN["sample_seed"]),
    "GPR_both": np.unique(lookup_GPR_both["seed"]),
    "GPR_traj": np.unique(lookup_GPR_traj["seed"]),
}
print(seeds)

# -----------------------------------------------
# Error computations
# -----------------------------------------------

# >> PaiNN << Error computation

# This will store all of the errors for [data fraction, training subset seed, ensemble rank, prediction point]
# Error = True - Predicted
all_errors_PaiNN = np.zeros(
    (fracs["PaiNN"].size, seeds["PaiNN"].size, painn_ranks.size, df.shape[0])
)

E_true = df["E_barrier"].to_numpy()

# Iterate over each training subset fraction, each seed and each ensemble rank
for i_frac, frac in enumerate(fracs["PaiNN"]):
    for i_seed, seed in enumerate(seeds["PaiNN"]):
        for i_rank, rank in enumerate(painn_ranks):

            # get the run that corresponds to the loop index values, i.e. data fraction, rank and seed
            run_hash = lookup_PaiNN[
                (
                    np.isclose(lookup_PaiNN["frac"], frac)
                    & (lookup_PaiNN["rank"] == rank)
                    & (lookup_PaiNN["sample_seed"] == seed)
                )
            ]
            assert (
                run_hash.shape[0] == 1
            )  # should be exactly one which fulfils criteria

            run_hash = run_hash.iloc[0].name
            run_column = f"E_PaiNN={run_hash}"

            # Get the predictions based on the found run id
            E_predict = df[run_column].to_numpy()

            # get errors and corresponding MAE
            Errors = E_true.copy() - E_predict
            mae = MAE(errs=Errors)

            # save errors
            all_errors_PaiNN[i_frac, i_seed, i_rank] = Errors

            print(f"Frac: {frac:.2f}, Seed: {seed}, Rank: {rank}, MAE: {mae:.2f}")


def get_maes_PaiNN(errors, df, d_max, origin):
    """Calculate the MAE and its standard deviation for individual runs as well as the ensemble runs"""

    # if there is a HAT distance cut-off filter, then only use those errors
    mask = (df.d <= d_max).to_numpy()
    if origin is not None:
        mask = mask * (df.origin == origin).to_numpy()
    errors = errors[:, :, :, mask]

    # Find the MAE for each individual run
    mae_per_run = np.mean(np.abs(errors), axis=3)

    # Find the mean and std of MAE for each training subset fraction, independent of ensemble rank
    mae_per_frac__mean = np.mean(mae_per_run, axis=(1, 2))
    mae_per_frac__std = np.std(mae_per_run, axis=(1, 2), ddof=1)
    print(f"MAE per frac \n{mae_per_frac__mean}\n{mae_per_frac__std}")

    # Find the MAE of each ensemble
    mae_per_ensemble = np.mean(np.abs(np.mean(errors, axis=2)), axis=2)
    # mean and std of MAE of each ensemble by training fraction
    mae_per_frac_ensemble__mean = np.mean(mae_per_ensemble, axis=1)
    mae_per_frac_ensemble__std = np.std(mae_per_ensemble, axis=1, ddof=1)
    print(
        f"Ensemble MAE per frac \n{mae_per_frac_ensemble__mean}\n{mae_per_frac_ensemble__std}"
    )

    return (
        mae_per_frac__mean,
        mae_per_frac__std,
        mae_per_frac_ensemble__mean,
        mae_per_frac_ensemble__std,
    )


def select_fracs(fracs, fracs_all):
    """Create mask to select only training fractions which were used in method"""
    return np.isin(fracs_all, fracs)


# Create a statistics table for the mean and std of MAE values
df_stats = pd.DataFrame(data=fracs["all"], columns=["frac"])
n_train_all = (pd.read_pickle(path_atoms).split == "train").sum()
frac2n = {f"{frac:.4f}": int(n_train_all * frac) * 2 for frac in fracs["all"]}
df_stats["n_train"] = [frac2n[f"{frac:.4f}"] for frac in df_stats["frac"]]

# only use those fractions which PaiNN used
sel = select_fracs(fracs["PaiNN"], fracs["all"])

# Get MAE statistics for HAT distance cut-off at 2 angstrom
statistics_PaiNN = get_maes_PaiNN(all_errors_PaiNN, df, 2, "traj")
df_stats.loc[sel, "PaiNN Individual MEAN 2A traj"] = statistics_PaiNN[0]
df_stats.loc[sel, "PaiNN Individual STD 2A traj"] = statistics_PaiNN[1]
df_stats.loc[sel, "PaiNN Ensemble MEAN 2A traj"] = statistics_PaiNN[2]
df_stats.loc[sel, "PaiNN Ensemble STD 2A traj"] = statistics_PaiNN[3]

# Get MAE statistics for HAT distance cut-off at 3 angstrom
statistics_PaiNN = get_maes_PaiNN(all_errors_PaiNN, df, 3, "traj")
df_stats.loc[sel, "PaiNN Individual MEAN 3A traj"] = statistics_PaiNN[0]
df_stats.loc[sel, "PaiNN Individual STD 3A traj"] = statistics_PaiNN[1]
df_stats.loc[sel, "PaiNN Ensemble MEAN 3A traj"] = statistics_PaiNN[2]
df_stats.loc[sel, "PaiNN Ensemble STD 3A traj"] = statistics_PaiNN[3]

# Get MAE statistics for HAT distance for no cut-off
statistics_PaiNN = get_maes_PaiNN(all_errors_PaiNN, df, np.inf, "traj")
df_stats.loc[sel, "PaiNN Individual MEAN all traj"] = statistics_PaiNN[0]
df_stats.loc[sel, "PaiNN Individual STD all traj"] = statistics_PaiNN[1]
df_stats.loc[sel, "PaiNN Ensemble MEAN all traj"] = statistics_PaiNN[2]
df_stats.loc[sel, "PaiNN Ensemble STD all traj"] = statistics_PaiNN[3]

# ---------------------------------------------------------------------
# >> GPR Both (training: traj.&synth.) << Error computation


def get_maes_GPR(errors, df, d_max, origin):
    """Calculate the MAE and its standard deviation for individual runs"""

    # if there is a HAT distance cut-off filter, then only use those errors
    mask = (df.d <= d_max).to_numpy()
    if origin is not None:
        mask = mask * (df.origin == origin).to_numpy()
    errors = errors[:, :, mask]

    # Find the MAE for each individual run per seed
    mae_per_seed = np.mean(np.abs(errors), axis=2)

    # Find the mean and std of MAE for each training subset fraction
    mae_per_frac__mean = np.mean(mae_per_seed, axis=1)
    mae_per_frac__std = np.std(mae_per_seed, axis=1, ddof=1)

    # sanity check
    assert mae_per_frac__mean.ndim == 1 and mae_per_frac__std.ndim == 1

    return mae_per_frac__mean, mae_per_frac__std


# This will store all of the errors for [data fraction, training subset seed, prediction point]
# Error = True - Predicted
all_errors_GPR_both = np.zeros(
    (fracs["GPR_both"].size, seeds["GPR_both"].size, df.shape[0])
)

E_true = df["E_barrier"].to_numpy().copy()

# Iterate over each training subset fraction and each seed
for i_frac, frac in enumerate(fracs["GPR_both"]):
    for i_seed, seed in enumerate(seeds["GPR_both"]):

        # get the run that corresponds to the loop index values, i.e. data fraction and seed
        run_hash = lookup_GPR_both[
            (
                np.isclose(lookup_GPR_both["frac"], frac)
                & (lookup_GPR_both["seed"] == seed)
            )
        ]

        assert run_hash.shape[0] == 1  # should be exactly one which fulfils criteria

        run_hash = run_hash.iloc[0].name
        run_column = f"E_GPR_both={run_hash}"

        # Get the predictions based on the found run id
        E_predict = df[run_column].to_numpy()

        # get errors and corresponding MAE
        Errors = E_true.copy() - E_predict
        mae = MAE(errs=Errors)

        # save errors
        all_errors_GPR_both[i_frac, i_seed] = Errors

        print(f"Frac: {frac:.2f}, Seed: {seed}, MAE: {mae:.2f}")


# Add to statistics table the mean and std of MAE values
statistics_GPR_both = get_maes_GPR(all_errors_GPR_both, df, 2, "traj")

# only use those fractions which method used
sel = select_fracs(fracs["GPR_both"], fracs["all"])

# Get MAE statistics for HAT distance cut-off at 2 angstrom
df_stats.loc[sel, "GPR Both MEAN 2A traj"] = statistics_GPR_both[0]
df_stats.loc[sel, "GPR Both STD 2A traj"] = statistics_GPR_both[1]

# Get MAE statistics for HAT distance cut-off at 3 angstrom
statistics_GPR_both = get_maes_GPR(all_errors_GPR_both, df, 3, "traj")
df_stats.loc[sel, "GPR Both MEAN 3A traj"] = statistics_GPR_both[0]
df_stats.loc[sel, "GPR Both STD 3A traj"] = statistics_GPR_both[1]

# Get MAE statistics for HAT distance for no cut-off
statistics_GPR_both = get_maes_GPR(all_errors_GPR_both, df, np.inf, "traj")
df_stats.loc[sel, "GPR Both MEAN all traj"] = statistics_GPR_both[0]
df_stats.loc[sel, "GPR Both STD all traj"] = statistics_GPR_both[1]

# ---------------------------------------------------------------------
# >> GPR Traj (training: traj. only) << Error computation

# This will store all of the errors for [data fraction, training subset seed, prediction point]
# Error = True - Predicted
all_errors_GPR_traj = np.zeros(
    (fracs["GPR_traj"].size, seeds["GPR_traj"].size, df.shape[0])
)

E_true = df["E_barrier"].to_numpy().copy()

# Iterate over each training subset fraction and each seed
for i_frac, frac in enumerate(fracs["GPR_traj"]):
    for i_seed, seed in enumerate(seeds["GPR_traj"]):

        # get the run that corresponds to the loop index values, i.e. data fraction and seed
        run_hash = lookup_GPR_traj[
            (
                np.isclose(lookup_GPR_traj["frac"], frac)
                & (lookup_GPR_traj["seed"] == seed)
            )
        ]

        assert run_hash.shape[0] == 1

        run_hash = run_hash.iloc[0].name
        run_column = f"E_GPR_traj={run_hash}"

        # Get the predictions based on the found run id
        E_predict = df[run_column].to_numpy()

        # get errors and corresponding MAE
        Errors = E_true.copy() - E_predict
        mae = MAE(errs=Errors)

        # save errors
        all_errors_GPR_traj[i_frac, i_seed] = Errors

        print(f"Frac: {frac:.2f}, Seed: {seed}, MAE: {mae:.2f}")


# Add to statistics table the mean and std of MAE values
statistics_GPR_traj = get_maes_GPR(all_errors_GPR_traj, df, 2, "traj")

# only use those fractions which method used
sel = select_fracs(fracs["GPR_traj"], fracs["all"])

# Get MAE statistics for HAT distance cut-off at 2 angstrom
df_stats.loc[sel, "GPR Traj MEAN 2A traj"] = statistics_GPR_traj[0]
df_stats.loc[sel, "GPR Traj STD 2A traj"] = statistics_GPR_traj[1]

# Get MAE statistics for HAT distance cut-off at 3 angstrom
statistics_GPR_traj = get_maes_GPR(all_errors_GPR_traj, df, 3, "traj")
df_stats.loc[sel, "GPR Traj MEAN 3A traj"] = statistics_GPR_traj[0]
df_stats.loc[sel, "GPR Traj STD 3A traj"] = statistics_GPR_traj[1]

# Get MAE statistics for HAT distance for no cut-off
statistics_GPR_traj = get_maes_GPR(all_errors_GPR_traj, df, np.inf, "traj")
df_stats.loc[sel, "GPR Traj MEAN all traj"] = statistics_GPR_traj[0]
df_stats.loc[sel, "GPR Traj STD all traj"] = statistics_GPR_traj[1]

# ---------------------------------------------------
# Overall MAE statistics for all methods
print(df_stats)
# ---------------------------------------------------

# Final GPR parameters statistics


def create_parameters_df(log):
    """Create a table with all final parameters of the GPR optimization indexed by run hash"""
    run_hashes = list(log.keys())
    for s in ["hashid", "frac", "seed", "final_params"]:
        run_hashes.remove(s)

    ds = []
    for run_hash in run_hashes:  # iteratively collect the final optimized parameters
        d = log[run_hash]["final_params"]
        d["frac"] = log[run_hash]["n_train"]
        d["seed"] = log[run_hash]["seed"]
        d["run_hash"] = run_hash
        ds.append(pd.json_normalize(d, sep="_"))
    # merge all final parameters into one table
    df_parameters = pd.concat(ds, axis=0).set_index("run_hash")
    return df_parameters


# Create tables which store the final optimized GPR parameters for each run
df_parameters_GPR_both = create_parameters_df(x_GPR_both["log"].item())
df_parameters_GPR_traj = create_parameters_df(x_GPR_traj["log"].item())


def get_average_parameters(df_parameters, n_seeds):
    """Computes the mean and std of the final optimized GPR parameters"""

    # list of all parameter names
    param_names = [
        param_name
        for param_name in df_parameters.columns
        if param_name not in ["seed", "frac"]
    ]

    # dict. to store all of the parameter values
    results = (
        {"frac": []}
        | {f"{param_name}_mean": [] for param_name in param_names}
        | {f"{param_name}_std": [] for param_name in param_names}
    )

    # group the parameters by the training subset fraction used
    fracs_group = df_parameters.groupby("frac")
    for (
        frac,
        frac_group,
    ) in fracs_group:  # iterate over all training fraction sub tables
        assert (
            frac_group.shape[0] == n_seeds
        )  # should be as many values for parameters as there are seeds

        results["frac"].append(frac)
        # for each parameter compute mean and std
        for param_name in param_names:
            # use absolute value of each parameter, since all are squared and therefore negative values do also occur
            results[f"{param_name}_mean"].append(
                frac_group[param_name].abs().mean().round(7)
            )
            results[f"{param_name}_std"].append(
                np.std(frac_group[param_name].abs(), ddof=1).round(7)
            )

    # create dataframe from the results
    df_params_stats = pd.DataFrame.from_dict(results)

    return df_params_stats


# compute the mean and std of all parameters by training fraction
df_params_GPR_both = get_average_parameters(
    df_parameters_GPR_both, seeds["GPR_both"].size
)
df_params_GPR_traj = get_average_parameters(
    df_parameters_GPR_traj, seeds["GPR_traj"].size
)

# get the final mean and std of parameters when using all training data (for GPR Traj this means all trajectory data)
final_parameters_both = df_params_GPR_both[df_params_GPR_both.frac == 1.0].iloc[0]
final_parameters_traj = df_params_GPR_traj[
    df_params_GPR_traj.frac == fracs["GPR_traj"].max()
].iloc[0]

print("Save")
df_stats.to_pickle("results/mae_scenarios.pkl")
lookup_PaiNN.to_pickle("results/lookup_PaiNN.pkl")
lookup_GPR_both.to_pickle("results/lookup_GPR_both.pkl")
lookup_GPR_traj.to_pickle("results/lookup_GPR_traj.pkl")
df.to_pickle("results/all_predictions.pkl")
df_params_GPR_both.to_pickle("results/gpr_both_params_stats.pkl")
df_params_GPR_traj.to_pickle("results/gpr_traj_params_stats.pkl")


# Print the final GPR parameters
def display_with_significant_digits(mean, std):
    significant_place_std = f"{np.max((-np.floor(np.log10(std)), 1)):.0f}"
    return f"{mean:.{significant_place_std}f} ± {std:.{significant_place_std}f}"


print("Final parameters SOAP GPR\n")
print("Training on all data:")
print(
    f'λₛ = {display_with_significant_digits(final_parameters_both["soap_s_0.0_lambda_mean"], final_parameters_both["soap_s_0.0_lambda_std"])}\n'
    f'λₘ = {display_with_significant_digits(final_parameters_both["soap_s_5.0_lambda_mean"], final_parameters_both["soap_s_5.0_lambda_std"])}\n'
    f'λₑ = {display_with_significant_digits(final_parameters_both["soap_s_10.0_lambda_mean"], final_parameters_both["soap_s_10.0_lambda_std"])}\n'
    f'λd = {display_with_significant_digits(final_parameters_both["d_lambda_mean"], final_parameters_both["d_lambda_std"])}\n'
    f'σ = {display_with_significant_digits(final_parameters_both["general_sigma_mean"], final_parameters_both["general_sigma_std"])}\n'
    f'g = {display_with_significant_digits(final_parameters_both["general_nugget_mean"], final_parameters_both["general_nugget_std"])}\n'
)
print("")
print("Training on trajectory data only:")
print(
    f'λₛ = {display_with_significant_digits(final_parameters_traj["soap_s_0.0_lambda_mean"], final_parameters_traj["soap_s_0.0_lambda_std"])}\n'
    f'λₘ = {display_with_significant_digits(final_parameters_traj["soap_s_5.0_lambda_mean"], final_parameters_traj["soap_s_5.0_lambda_std"])}\n'
    f'λₑ = {display_with_significant_digits(final_parameters_traj["soap_s_10.0_lambda_mean"], final_parameters_traj["soap_s_10.0_lambda_std"])}\n'
    f'λd = {display_with_significant_digits(final_parameters_traj["d_lambda_mean"], final_parameters_traj["d_lambda_std"])}\n'
    f'σ = {display_with_significant_digits(final_parameters_traj["general_sigma_mean"], final_parameters_traj["general_sigma_std"])}\n'
    f'g = {display_with_significant_digits(final_parameters_traj["general_nugget_mean"], final_parameters_traj["general_nugget_std"])}\n'
)

# MAE
mae_train_all = df_stats[df_stats.frac == 1.0].iloc[0]
mae_train_traj = df_stats[df_stats.frac == fracs["GPR_traj"].max()].iloc[0]

# Test all traj
print("Test: ALL TRAJ")
print(
    f'SOAP (train all): {display_with_significant_digits(mae_train_all["GPR Both MEAN all traj"], mae_train_all["GPR Both STD all traj"])} kcal/mol'
)
print(
    f'SOAP (train traj): {display_with_significant_digits(mae_train_traj["GPR Traj MEAN all traj"], mae_train_traj["GPR Traj STD all traj"])} kcal/mol'
)
print(
    f'PaiNN (individual): {display_with_significant_digits(mae_train_all["PaiNN Individual MEAN all traj"], mae_train_all["PaiNN Individual STD all traj"])} kcal/mol'
)
print(
    f'PaiNN (ensemble): {display_with_significant_digits(mae_train_all["PaiNN Ensemble MEAN all traj"], mae_train_all["PaiNN Ensemble STD all traj"])} kcal/mol'
)
print("")

# Test all traj 3A
print("Test: <3A TRAJ")
print(
    f'SOAP (train all): {display_with_significant_digits(mae_train_all["GPR Both MEAN 3A traj"], mae_train_all["GPR Both STD 3A traj"])} kcal/mol'
)
print(
    f'SOAP (train traj): {display_with_significant_digits(mae_train_traj["GPR Traj MEAN 3A traj"], mae_train_traj["GPR Traj STD 3A traj"])} kcal/mol'
)
print(
    f'PaiNN (individual): {display_with_significant_digits(mae_train_all["PaiNN Individual MEAN 3A traj"], mae_train_all["PaiNN Individual STD 3A traj"])} kcal/mol'
)
print(
    f'PaiNN (ensemble): {display_with_significant_digits(mae_train_all["PaiNN Ensemble MEAN 3A traj"], mae_train_all["PaiNN Ensemble STD 3A traj"])} kcal/mol'
)
print("")

# Test all traj 2A
print("Test: <2A TRAJ")
print(
    f'SOAP (train all): {display_with_significant_digits(mae_train_all["GPR Both MEAN 2A traj"], mae_train_all["GPR Both STD 2A traj"])} kcal/mol'
)
print(
    f'SOAP (train traj): {display_with_significant_digits(mae_train_traj["GPR Traj MEAN 2A traj"], mae_train_traj["GPR Traj STD 2A traj"])} kcal/mol'
)
print(
    f'PaiNN (individual): {display_with_significant_digits(mae_train_all["PaiNN Individual MEAN 2A traj"], mae_train_all["PaiNN Individual STD 2A traj"])} kcal/mol'
)
print(
    f'PaiNN (ensemble): {display_with_significant_digits(mae_train_all["PaiNN Ensemble MEAN 2A traj"], mae_train_all["PaiNN Ensemble STD 2A traj"])} kcal/mol'
)
