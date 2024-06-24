"""
Main script for GPR training and predictions using SOAP features
"""

import jax
from jax import numpy as jnp
import numpy as np
from functools import partial
import time
from scipy.optimize import minimize
import pandas as pd
from pathlib import Path
import uuid
from pprint import pprint, pformat
from sklearn import preprocessing
from sklearn.metrics import mean_absolute_error
import sys

sys.path.append(str(Path.cwd().parent.resolve()))
import general.GPR_helper as gpr


def get_K(params, D2):
    """Create kernel matrix from distance matrices"""

    # Each kernel function takes as argument the relevant distance matrix
    # (either transition distance difference of SOAP feature difference)

    K = (
        gpr.SE_paper(params["d"], D2["d"])
        * gpr.SE_paper(params["soap"]["s_0.0"], D2["soap"]["s_0.0"])
        * gpr.SE_paper(params["soap"]["s_5.0"], D2["soap"]["s_5.0"])
        * gpr.SE_paper(params["soap"]["s_10.0"], D2["soap"]["s_10.0"])
    )
    return K


# ----------------- Predictions -------------------


def predict(*, Y_inference, D2, params, config):
    """Calculate predictions"""
    print("Starting prediction")

    # Compute K(train, train)
    K_inference = get_K(params, D2["train_all"])
    K_inference = gpr.K_transform_general(K_inference, params, config["run_config"])

    # Avoid costly and unstable naive matrix inversion with Cholesky decomposition
    print("Cholesky decomposition")
    _, alpha = gpr.get_diagL_a(K_inference, Y_inference)

    # Compute K(train, test)
    K_n_star = get_K(params, D2["train_test"])
    K_n_star = (params["general"]["sigma"] ** 2) * K_n_star

    # GPR mean prediction
    print("Predict mean")
    mean = gpr.predict_mean(K_n_star, alpha)

    if config["predict_uncertainty"]:
        print("Get uncertainty")

        # Cholesky decomposition as before
        L = jax.scipy.linalg.cholesky(K_inference, lower=True)
        # solve does (L @ L.T)^{-1} @ x = y -> x = solve(L, y)
        beta = L.T @ jax.scipy.linalg.cho_solve((L, True), K_n_star)

        # Compute K(test, test)
        K_star = get_K(params, D2["test_test"])
        K_star = gpr.K_transform_general(K_star, params, config["run_config"])

        # Conditional variance is on the diagonal
        var = (jnp.diagonal(K_star) - jnp.sum(beta * beta, axis=0))[..., None]
    else:
        var = np.nan

    return mean, var


@partial(jax.jit, static_argnames=["params_treedef"])
def NLL_and_grad(params_opt_array, params_treedef, D_split, Y_train, config):
    """Calculate negative log-likelihood. Using JAX the gradient can also be calculated automatically"""
    # Convert parameter array back to parameter dictionary
    params = jax.tree_util.tree_unflatten(params_treedef, params_opt_array)
    # Get the covariance matrix of the training batch
    K = get_K(params, D_split)
    # Apply general transformation: s**2*(K+g)+jitter
    K = gpr.K_transform_general(K, params, config)

    return gpr.NLL(K, Y_train)


def loss(params_opt_array, params_treedef, D_splits, Y_train_splits, config):
    """Calculate loss. This is done with training batches one at a time."""
    NLL = 0  # Loss value
    GRAD = jnp.zeros(len(params_opt_array))  # Collect gradient vector of all parameters

    # Here the loss is a summation of the NLL of each batch: Loss = Σ_{i=1}^{N} NLL_i

    # The gradient is then also the sum:
    # ∂Loss(θ_1, θ_2, ..., θ_n)/∂θ_i =
    # ∂( Σ_{i=1}^{N} NLL_i(θ_1, θ_2, ..., θ_n) )/∂θ_i =
    # Σ_{i=1}^{N} ∂NLL_i(θ_1, θ_2, ..., θ_n)/∂θ_i,
    # where θ_k is the k^th parameter and NLL_i is the neg. log-likelihood of batch i

    # Calculate the loss block-wise at a time. For this transfer it to the GPU and afterwards back to CPU
    for i in range(len(D_splits)):  # Loop through all batches
        tic = time.time()  # Start time to access compute time during optimization
        # Load the batch, i.e. distance matrices of the training batch, to GPU memory
        D_splits[i] = {
            k: jax.device_put(D, config["gpu"]) for k, D in D_splits[i].items()
        }
        # Calculate the neg. log-likelihood, i.e. loss, and gradients
        nll, grad = jax.value_and_grad(NLL_and_grad)(
            params_opt_array,
            params_treedef,
            D_splits[i],
            Y_train_splits[i],
            config["run_config"],
        )
        # Load the batch, i.e. distance matrices of the training batch, to CPU memory
        D_splits[i] = {
            k: jax.device_put(D, config["cpu"]) for k, D in D_splits[i].items()
        }

        # Add loss and gradients
        NLL += nll
        GRAD += jnp.array(grad)

        print(f"Iteration {i} time: {time.time()-tic:.3f}s")

    return float(NLL), GRAD.tolist()


def main(config):

    # The default jax default memory location (cpu or gpu) and floating point precision
    gpr.set_jax(config["accelerator"], float64=config["float64"])
    cpu = jax.devices("cpu")[0]
    gpu = jax.devices("gpu")[0]
    config["cpu"] = cpu
    config["gpu"] = gpu

    # Save the run id to the config for later identification
    config["hashid"] = str(uuid.uuid4())

    config["plot_save"] = config["data_dir"] + config["plot_filename"]
    config["config_save"] = config["data_dir"] + config["config_filename"]
    config["parameters_save"] = config["data_dir"] + config["parameters_name"]

    print(f"Configuration: {pformat(config, indent=4)}", flush=True)

    # Start with the initial parameters
    params = config["params_init"]

    # Set seed for reproducibility
    np.random.seed(config["seed"])

    # Load the data and preprocess, like split into validation set or use just subset for training
    print("Load")
    df = pd.read_pickle(config["df"])
    df = gpr.prepare_df(df, config)

    # Sanity check later
    tmp_E = [
        df.iloc[0].E_forward.copy(),
        df.iloc[0].E_backward.copy(),
        df.iloc[1].E_forward.copy(),
        df.iloc[1].E_backward.copy(),
        df.iloc[-2].E_forward.copy(),
        df.iloc[-2].E_backward.copy(),
        df.iloc[-1].E_forward.copy(),
        df.iloc[-1].E_backward.copy(),
    ]

    types = df.dtypes.to_dict()
    # Original dataframe has for every row a forward and backward barrier.
    # We want each row to have one target value,
    # so double the dataframe with every even row having the forward barriers and every odd row the backward
    df = pd.DataFrame(np.repeat(df.values, 2, axis=0), columns=df.columns)
    df.loc[0::2, "E_barrier"] = df.iloc[0::2]["E_forward"]
    df.loc[0::2, "direction"] = "forward"
    df.loc[1::2, "E_barrier"] = df.iloc[1::2]["E_backward"]
    df.loc[1::2, "direction"] = "backward"
    # Correct types and drop unnecessary columns
    df = df.astype(types)
    df.drop(columns=["E_forward", "E_backward", "atoms", "r_H", "E"], inplace=True)

    # Sanity check from before
    assert (
        np.isclose(df.iloc[0].E_barrier, tmp_E[0])
        and np.isclose(df.iloc[1].E_barrier, tmp_E[1])
        and np.isclose(df.iloc[2].E_barrier, tmp_E[2])
        and np.isclose(df.iloc[3].E_barrier, tmp_E[3])
        and np.isclose(df.iloc[-4].E_barrier, tmp_E[-4])
        and np.isclose(df.iloc[-3].E_barrier, tmp_E[-3])
        and np.isclose(df.iloc[-2].E_barrier, tmp_E[-2])
        and np.isclose(df.iloc[-1].E_barrier, tmp_E[-1])
    )

    # Sanity check directions and barriers
    assert (
        df.iloc[0].direction == "forward"
        and df.iloc[1].direction == "backward"
        and df.iloc[-2].direction == "forward"
        and df.iloc[-1].direction == "backward"
    )

    # Assign barrier values to Y and HAT transition distances to d
    print("Get data")
    Y = df.E_barrier.to_numpy().astype(float)[..., None]
    d = df.d.to_numpy()[..., None]

    assert d.ndim == 2

    # Create boolean filter for training and test conditions as well as associated indices
    train_criteria, test_criteria = gpr.filter(config, df)
    train_indices = df[train_criteria].index.to_numpy()
    test_indices = df[test_criteria].index.to_numpy()

    # Sanity checks concerning the filter criteria
    assert np.allclose(df.index.to_numpy(), np.arange(df.shape[0]))
    assert (
        df.loc[train_indices].split.unique()[0] == "train"
        and df.loc[test_indices].split.unique()[0] == "test"
    )
    assert (
        config["test_d_bounds"][0] <= df.loc[test_indices].d.min()
        and df.loc[test_indices].d.max() <= config["test_d_bounds"][1]
    )
    assert (
        config["train_d_bounds"][0] <= df.loc[train_indices].d.min()
        and df.loc[train_indices].d.max() <= config["train_d_bounds"][1]
    )
    if config["test_origin"]:
        assert df.loc[test_indices].origin.unique()[0] == config["test_origin"]
    else:
        assert (
            df.loc[test_indices].origin.unique().size == 2
            and "traj" in df.loc[test_indices].origin.unique()
            and "synth" in df.origin.unique()
        )
    if config["train_origin"]:
        assert df.loc[train_indices].origin.unique()[0] == config["train_origin"]
    else:
        assert (
            df.loc[train_indices].origin.unique().size == 2
            and "traj" in df.loc[train_indices].origin.unique()
            and "synth" in df.origin.unique()
        )

    assert df[train_criteria].transition_hash.equals(
        df.loc[train_indices].transition_hash
    ) and df.loc[train_indices].transition_hash.equals(
        df.transition_hash[train_indices]
    )
    assert df[test_criteria].transition_hash.equals(
        df.loc[test_indices].transition_hash
    ) and df.loc[test_indices].transition_hash.equals(df.transition_hash[test_indices])

    # Number of batches
    n_splits = int(config["run_config"]["n_split"])
    # Split all training indices into equally sized batches. Last training values might not be included
    train_indices_splits = np.split(
        train_indices[: n_splits * (train_indices.size // n_splits)], n_splits
    )
    # All training indices part of some batch
    all_train_indices = np.array(train_indices_splits).flatten()
    print(f"Batches of sizes: {[s.size for s in train_indices_splits]}")

    # Y_train contains ALL training data, so also those not included in a batch
    Y_train = Y[train_indices, :]
    # Y_test is never batched, so always has all test values
    Y_test = Y[test_indices, :]

    # Get scaler to normalize Y training and test date based on training data
    Y_scaler = preprocessing.StandardScaler(with_mean=True, with_std=True).fit(
        Y[all_train_indices, :]
    )
    print(f"Y Scaler: scale={Y_scaler.scale_}, mean={Y_scaler.mean_} ")
    Y = Y_scaler.transform(Y)

    # Get scaler to normalize d training and test date based on training data
    d_scaler = preprocessing.StandardScaler(with_mean=True, with_std=True).fit(
        d[all_train_indices, :]
    )
    print(f"d Scaler: scale={d_scaler.scale_}, mean={d_scaler.mean_} ")
    d = d_scaler.transform(d)

    print("n_train:", Y_train.shape[0], "n_test: ", Y_test.shape[0])
    assert (
        Y_train.ndim == 2 and Y_test.ndim == 2 and Y.ndim == 2
    ), "Dimensions of Y not as expected!"

    # Batches of training Y (barrier) data as list
    Y_train_splits = [
        Y[train_indices_split, :] for train_indices_split in train_indices_splits
    ]

    # Create a dictionary that will store all the distance matrices for training & testing
    # This includes SOAP distance matrices and HAT transition distance matrices
    soap_names = list(config["soaps"].keys())
    D2 = {
        "train": [
            {"soap": {s: None for s in soap_names} | {"d": None}}
            for _ in range(n_splits)
        ],
        "train_test": {"soap": {s: None for s in soap_names} | {"d": None}},
        "test_test": {"soap": {s: None for s in soap_names} | {"d": None}},
        "train_all": {"soap": {s: None for s in soap_names} | {"d": None}},
    }

    print("Calculate D")
    # Compute the HAT transition distance matrix and save it to cpu memory
    D2_d = jax.device_put(gpr.get_D(X=d, exponent=2), cpu)

    # Now save the relevant sub matrices to the dictionary
    D2["train_all"]["d"] = jax.device_put(
        D2_d[np.ix_(all_train_indices, all_train_indices)], cpu
    )
    D2["train_test"]["d"] = jax.device_put(
        D2_d[np.ix_(all_train_indices, test_indices)], cpu
    )
    if config["predict_uncertainty"]:
        D2["test_test"]["d"] = jax.device_put(
            D2_d[np.ix_(test_indices, test_indices)], cpu
        )

    for i in range(len(train_indices_splits)):
        D2["train"][i]["d"] = jax.device_put(
            D2_d[np.ix_(train_indices_splits[i], train_indices_splits[i])], cpu
        )

    transition_hashes = df.transition_hash.to_numpy()
    directions = df.direction.to_numpy()

    # Load precomputed SOAP distance matrices and add them to the dictionary
    for soap_name in soap_names:
        gpr.set_D2_soap(
            str(config["soap_dir"] / config["soaps"][soap_name]),
            D2,
            train_indices_splits,
            test_indices,
            transition_hashes,
            directions,
            cpu,
            # ! Important this will apply an exponent to all distances, since this expected in the SE kernel. This saves
            # repeated computation, but may need to be changed for a different kernel
            name=soap_name,
        )
    if len(soap_names) == 0:
        print("Skipping soap measure since not in parameters!")

    print(f"Y Batches of sizes: {[s.size for s in Y_train_splits]}")

    # Parameters will have to be converted to array and later back to a dictionary, therefore save the structure
    params_treedef = jax.tree_util.tree_structure(params)

    if config["minimise_nll"]:

        def current_minimise_state(xk):
            """Returns current parameter values at minimization iteration"""
            params = jax.tree_util.tree_unflatten(params_treedef, xk)
            print(f"Currently with parameters: {params}", flush=True)

        # Initial parameters
        x0 = np.array(jax.tree_util.tree_leaves(params))
        print("x0:")
        pprint(params)

        tic = time.time()
        # Start minimization
        res = minimize(
            loss,
            x0,
            method="L-BFGS-B",
            jac=True,
            args=(params_treedef, D2["train"], Y_train_splits, config),
            options={"iprint": 1},
            callback=current_minimise_state,
        )
        t_min = time.time() - tic
        # Get final parameters in dict form
        params = jax.tree_util.tree_unflatten(params_treedef, res.x)

        print(f"Minimisation Results in {t_min:.3f}s:\n{res}")
        print(
            "\n******************************************************************************************"
        )
        pprint(params)
        print(f"With NLL: {res.fun}")
        print(
            "******************************************************************************************"
        )
    else:
        print("NO MINIMISATION as requested")

    print(f"Using parameters: {pformat(params, indent=4)}")

    # Make prediction on the test (or validation) set
    print("Predict Ys")
    Y_inference = jnp.array(Y_train_splits).flatten()[..., None]
    Y_predict, var = predict(
        Y_inference=Y_inference, D2=D2, params=params, config=config
    )

    # Transform mean back to original scale
    Y_predict = Y_scaler.inverse_transform(Y_predict)
    sigma_pred = np.sqrt(Y_scaler.scale_**2) * np.sqrt(var)

    # Print some example predictions and calculate the error
    print("[Y_predict, Y_test, Abs Diff]")
    print(
        np.hstack(
            [Y_predict[:5, :], Y_test[:5, :], np.abs(Y_predict[:5, :] - Y_test[:5, :])]
        )
    )
    MAE_barrier = gpr.get_MAE(Y_predict, Y_test)
    # Sanity check
    assert (
        abs(mean_absolute_error(Y_test, Y_predict) - MAE_barrier) < 1e-6
    ), f"{mean_absolute_error(Y_test, Y_predict)}, {MAE_barrier}"

    print("******************************************************")
    print(f" MAE barrier predictions: {MAE_barrier:.2f}")
    print("******************************************************")

    # Plot scatter plot of predictions vs true values and save
    Path(config["plot_save"]).unlink(missing_ok=True)
    gpr.plot_MAE(
        color="#00509d",
        Y_test=Y_test,
        Y_predict=Y_predict,
        MAE=MAE_barrier,
        config=config,
    )

    # Save final parameters if specified
    if config["save_parameters"]:
        np.savez(config["parameters_save"], params=params, hashid=config["hashid"])

    # Add configuration and result to log file
    with open(config["config_save"], "a+") as f:
        f.write(
            f"Config: {pformat(config, indent=4)}\n\n"
            f"Prediction parameters: {params} \n -> MAE: {MAE_barrier}\n\n"
            f"------------------------------------------------------------------------------------------------\n\n"
        )

    # Create dataframe with test predictions and perform varies sanity checks
    assert np.allclose(df.index.to_numpy(), np.arange(df.shape[0]))
    df_test = df.loc[test_indices].copy()
    assert np.unique(df_test.split.to_numpy())[0] == "test"
    assert np.sum(np.abs(df_test.E_barrier.to_numpy() - Y_test.flatten())) < 1e-6

    df_test["E_barrier_predict"] = Y_predict.flatten()
    df_test["E_barrier_predict_sigma"] = sigma_pred.flatten()
    assert (
        mean_absolute_error(
            df_test["E_barrier"].to_numpy(), df_test["E_barrier_predict"].to_numpy()
        )
        - MAE_barrier
    ) < 1e-6

    return df_test, MAE_barrier, params
