"""Script which calls the SOAP GRP with different fraction of training data and structure origin"""

from pathlib import Path
import numpy as np
from SOAP_GPR import main
import general.GPR_helper as gpr
import sys


######################################################################
# Important: Before, calculate the SOAP distance matrices using data/create_soap_distances.sh
######################################################################

# Script should be passed the rank of the process as a second argument
# One rank will either run SOAP GPR with all training data or only trajectory training data
assert len(sys.argv) == 2, "No rank passed"
rank = int(sys.argv[1])
print(f"Rank {rank} started")


cwd = Path(__file__).resolve().parent
project_root = cwd.parent

# Config file for the GPR SOAP method
config = dict(
    accelerator="cpu",  # Whether the default jax array lives in cpu or gpu memory
    save_parameters=False,  # Whether to save the final parameters to a file
    # Initial parameters used before optimization. If no optimization is done then this will be used for inference
    params_init={
        "general": {"nugget": np.nan, "sigma": np.nan},
        "soap": {
            "s_0.0": {"lambda": np.nan},
            "s_5.0": {"lambda": np.nan},
            "s_10.0": {"lambda": np.nan},
        },
        "d": {"lambda": np.nan},
    },
    # Precomputed SOAP distance matrix files at different steps of the HAT transition
    soaps={
        "s_0.0": "SOAP_dist_at_0.npz",
        "s_5.0": "SOAP_dist_at_5.npz",
        "s_10.0": "SOAP_dist_at_10.npz",
    },
    data_dir="results/",  # Folder where the results will be saved
    config_filename="config.txt",  # Config file name to save the run configuration and results to
    plot_filename="prediction.png",  # Name of prediction scatter plot
    df=project_root / "data" / "atoms.pkl",  # Dataframe with all data
    soap_dir=project_root
    / "data"
    / "soap",  # Directory where SOAP distance matrices are stored
    script=__file__,
    parameters_name="parameters.npz",  # File name where final paramters will be saved to
    float64=True,  # Floating point precision to use
    seed=np.nan,  # Seed for reproducibility
    test_d_bounds=[
        0.0,
        np.inf,
    ],  # HAT transition distance bounds to consider for test set
    test_origin=None,  # Origin of test set. None: synthetic and trajectory; "traj": trajectory; "synth": synthetic
    test_only_opt=False,  # Use only optimized test set
    train_d_bounds=[
        0.0,
        np.inf,
    ],  # HAT transition distance bounds to consider for training set
    train_origin="",  # Origin of test set. None: synthetic and trajectory; "traj": trajectory; "synth": synthetic
    train_only_opt=False,  # Use only optimized training set
    # Whether and how many training point to use. Float interpreted as fraction of all training set;
    # Int interpreted as absolute number; "all sequential" to use all training points without shuffling
    validation={"n_train": np.nan},
    # Jitter to add diagonal for numerical stability and number of batches to use for MLE minimization
    run_config=dict(jitter=1e-10, n_split=np.nan),
    test_production=True,  # Whether to use the real test set for testing. If False will use a validation set
    predict_uncertainty=True,  # Whether the uncertainty should be predicted
    minimise_nll=True,  # Whether MLE optimization should be performed to get best kernel parameters
    rank=rank,  # MPI rank of the run
)


def run_fractionals(n_trains, splits, config, save_name, seeds):
    """Run a SOAP GPR optimization with given training fractions and origin (all or only trajectory)."""

    # For each training fraction multiple seeds will be used to sample the training set randomly
    df_pred = None  # Save results in dataframe
    log = {
        "hashid": [],
        "frac": [],
        "seed": [],
        "final_params": [],
    }  # Save the metadata and results of the runs
    for i, n_train in enumerate(n_trains):  # For each training fraction
        for j, seed in enumerate(seeds):  # For each seed to sample the training subset
            print(f"n_train: {n_train}, seed: {seed}")

            # Adjust configuration for case
            config["seed"] = seed
            config["params_init"] = gpr.randomise_initial_parameters(
                config["params_init"], config["seed"]
            )
            config["validation"]["n_train"] = float(n_train)
            config["run_config"]["n_split"] = int(splits[i])
            config["minimise_nll"] = True

            # Run the GPR SOAP optimization
            # df_test saves the test predictions, MAE_barrier is the test MAE and params are the opt. kernel parameters
            df_test, MAE_barrier, params = main(config)

            # Save meta data
            hashid = config["hashid"]
            log[hashid] = {
                "n_train": n_train,
                "seed": seed,
                "final_params": params,
                "df_test": df_test,
            }
            # Each barrier gets a unique hash consisting of the reaction hash and the direction of the reaction
            df_test["hash_direction"] = [
                f'{df_test["transition_hash"].iloc[i]}_{df_test["direction"].iloc[i]}'
                for i in range(df_test.shape[0])
            ]
            print(n_train, seed, df_test, df_test.shape)

            # During the first run built a dataframe to save all results to
            if df_pred is None:
                df_pred = df_test.copy().set_index(["hash_direction"])
                df_pred.rename(
                    columns={
                        "E_barrier_predict": f"E_barrier_predict-{hashid}",
                        "E_barrier_predict_sigma": f"E_barrier_predict_sigma-{hashid}",
                    },
                    inplace=True,
                )
            # For all subsequent runs add the predictions to the results dataframe
            else:
                df_test = df_test.copy().set_index(["hash_direction"])

                # Sanity check that its the same test reactions
                assert (
                    len(
                        np.setdiff1d(df_test.index.to_numpy(), df_pred.index.to_numpy())
                    )
                    == 0
                )
                assert np.all(
                    np.isin(df_test.index.to_numpy(), df_pred.index.to_numpy())
                )

                # Sanity check that target barrier values are the same in both df_pred and df_test
                E_ref = np.array(
                    df_pred.loc[df_test.index]["E_barrier"].to_numpy(), dtype=np.float64
                )
                E_test = np.array(df_test["E_barrier"].to_numpy(), dtype=np.float64)
                assert np.allclose(E_ref, E_test)

                # Add predicted barriers and uncertainty to results dataframe
                df_pred.loc[df_test.index, f"E_barrier_predict-{hashid}"] = (
                    df_test["E_barrier_predict"].to_numpy().copy()
                )
                df_pred.loc[df_test.index, f"E_barrier_predict_sigma-{hashid}"] = (
                    df_test["E_barrier_predict_sigma"].to_numpy().copy()
                )

    # Save predictions and metadata
    np.savez(save_name, df_pred=df_pred, log=log)


# Seeds to use for each training fraction subset
seeds = [0, 1, 2, 3, 4, 5, 6, 7]

# Rank 0 trains on all training data (trajectory & synthetic)
if rank == 0:
    print("Run train all")
    # Total training data fraction and number of "batches". Not all training data may fit into memory
    n_trains_splits = np.array(
        [
            [0.01, 1],
            [0.02, 1],
            [0.03, 1],
            [0.04, 1],
            [0.05, 1],
            [0.1, 1],
            [0.2, 1],
            [0.3, 2],
            [0.4, 2],
            [0.5, 2],
            [0.6, 3],
            [0.7, 3],
            [0.8, 3],
            [0.9, 4],
            [1.0, 4],
        ]
    )[::-1, :]

    save_name = cwd / "results" / "GPR_predictions_train_both.npz"
    config["train_origin"] = None

    # Run SOAP GPR with config
    run_fractionals(
        n_trains=n_trains_splits[:, 0].tolist(),
        splits=n_trains_splits[:, 1].astype(int).tolist(),
        config=config,
        save_name=save_name,
        seeds=seeds,
    )

# Rank 1 trains on only trajectory data
elif rank == 1:
    print("Run train traj")
    # Total training data fraction and number of "batches". Not all training data may fit into memory
    n_trains_splits = np.array(
        [
            [0.01, 1],
            [0.02, 1],
            [0.03, 1],
            [0.04, 1],
            [0.05, 1],
            [0.1, 1],
            [0.2, 1],
            [0.3, 2],
            [0.4, 2],
            [0.5, 2],
            [0.5427, 2],
            # Max of available traj training points is 54.27% of all training data
        ]
    )[::-1, :]
    save_name = cwd / "results" / "GPR_predictions_train_traj.npz"
    config["train_origin"] = "traj"

    # Run SOAP GPR with config
    run_fractionals(
        n_trains=n_trains_splits[:, 0].tolist(),
        splits=n_trains_splits[:, 1].astype(int).tolist(),
        config=config,
        save_name=save_name,
        seeds=seeds,
    )
else:
    raise NotImplementedError(f"Provided rank not implemented: {rank}")

print("Done")
