"""Finds optimal kernel parameters sigma and the nugget term from the previously calculated kernel matrix"""

import sys
from pathlib import Path

sys.path.append(str(Path.cwd().parent.resolve()))
import general.GPR_helper as gpr

import jax
from jax import numpy as jnp
import numpy as np
from functools import partial
import time
from scipy.optimize import minimize
import pandas as pd
from pprint import pprint
import uuid
from sklearn import preprocessing
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


config = dict(
    K_graph="K.npy",  # Kernel matrix previously calculated with GraphDot
    df_load=f"collect_networks_diff.pkl",  # dataframe based on which K was calculated
    save_df_test=f"results/df_MGK_predictions.pkl",  # save predictions to this dataframe
    creation_config="creation_config.txt",  # configuration file log
    plot_save="results/prediction.png",  # prediction plot for sanity check
    run_config=dict(
        jitter=1e-10, n_split=3
    ),  # Jitter term and number of blocks to split kernel matrix
    minimise_nll=True,  # whether to optimize the parameter or only predict
    params_init={
        "general": {
            "nugget": 0.1,
            "sigma": 1.0,
        },
    },
    seed=0,  # random seed for reproducibility
    test_d_bounds=[
        0.0,
        np.inf,
    ],  # limits of transition distance allowed in the test set [Å]
    test_origin=None,  # Whether to only test trajectory, synthetic or all; "traj", "synth" or None respectively
    test_only_opt=False,  # Whether test cases should only contain optimized energies
    train_d_bounds=[
        0.0,
        np.inf,
    ],  # limits of transition distance allowed in the training set [Å]
    train_origin=None,  # Whether to only train trajectory, synthetic or all; "traj", "synth" or None respectively
    train_only_opt=False,  # Whether train cases should only contain optimized energies
    test_production=True,  # Test on final test set? Or use validation set instead
    float64=True,  # Floating point precision. float32 often fails
    accelerator="cpu",  # In which memory the default arrays should live
    validation={
        "n_train": "all_sequential"
    },  # How many training points to use. "all_sequential" to use all
    predict_uncertainty=False,  # Whether uncertainty estimate should be predicted
)


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
    NLL = 0
    GRAD = jnp.zeros(len(params_opt_array))

    # Here the loss is a summation of the NLL of each batch: Loss = Σ_{i=1}^{N} NLL_i

    # The gradient is then also the sum:
    # ∂Loss(θ_1, θ_2, ..., θ_n)/∂θ_i =
    # ∂( Σ_{i=1}^{N} NLL_i(θ_1, θ_2, ..., θ_n) )/∂θ_i =
    # Σ_{i=1}^{N} ∂NLL_i(θ_1, θ_2, ..., θ_n)/∂θ_i,
    # where θ_k is the k^th parameter and NLL_i is the neg. log-likelihood of batch i

    # Calculate the loss block-wise at a time. For this transfer it to the GPU and afterwards back to CPU
    for i in range(m := len(D_splits)):
        tic = time.time()
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
        params = jax.tree_util.tree_unflatten(params_treedef, params_opt_array)
        print(
            f"Split {i + 1} of {m} in {time.time() - tic:.3f}s ({params}, {nll:.2f} {grad})"
        )
        if np.isnan(nll):
            raise Exception("NLL is NaN")
    return float(NLL), GRAD.tolist()


def get_K(params, D2):
    # In this case the normalized covariance matrix is already precomputed
    return D2["K"]


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


np.random.seed(config["seed"])
params = config["params_init"]

# The default jax default memory location (cpu or gpu) and floating point precision
gpr.set_jax(config["accelerator"], float64=config["float64"])
cpu = jax.devices("cpu")[0]
gpu = jax.devices("gpu")[0]
config["cpu"] = cpu
config["gpu"] = gpu

# Save the run id to the config for later identification
config["hashid"] = str(uuid.uuid4())
pprint(config)

print("Load df")
df = pd.read_pickle(config["df_load"])
df = gpr.prepare_df(df, config)

# Load the precomputed normalized MGK covariance matrix
print("load K")
K = np.load(config["K_graph"])

assert K.shape[0] == df.shape[0]

# Assign barrier values to Y and HAT transition distances to d
Y = jnp.array(df["E_barrier"].to_numpy())[..., None]
d = df["d"].to_numpy()[..., None]

# Create boolean filter for training and test conditions as well as associated indices
assert np.allclose(np.arange(df.shape[0]), df.index.to_numpy())
train_criteria, test_criteria = gpr.filter(config, df)

train_indices = df[train_criteria].index.to_numpy()
test_indices = df[test_criteria].index.to_numpy()

print(f"All training points before splitting: {train_indices.size}")

Y_test = Y[test_indices, :]

# Number of batches
n_splits = int(config["run_config"]["n_split"])

# Split all training indices into equally sized batches. Last training values might not be included
train_indices_splits = np.split(
    train_indices[: n_splits * (train_indices.size // n_splits)], n_splits
)
# All training indices part of some batch
all_train_indices = np.array(train_indices_splits).flatten()
print(f"Batches of sizes: {[s.size for s in train_indices_splits]}")

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

# Batches of training Y (barrier) data as list
Y_train_splits = [
    Y[train_indices_split, :] for train_indices_split in train_indices_splits
]
n_training_with_splits = sum([y.size for y in Y_train_splits])

print("n_train:", n_training_with_splits, "n_test: ", Y_test.shape[0])
assert (
    Y[all_train_indices, :].ndim == 2 and Y_test.ndim == 2 and Y.ndim == 2
), "Dimensions of Y not as expected!"
assert not np.any(np.isin(test_indices, train_indices)) and np.all(
    np.isin(all_train_indices, train_indices)
)

# Create a dictionary that will store all the precomputed normalized covariance matrices for training & testing
D2 = {
    "train": [{"d": None, "K": None} for _ in range(n_splits)],
    "train_test": {"d": None, "K": None},
    "test_test": {"d": None, "K": None},
    "train_all": {"d": None, "K": None},
}


D2["train_all"]["K"] = jax.device_put(K[np.ix_(train_indices, train_indices)], cpu)
D2["train_test"]["K"] = jax.device_put(K[np.ix_(train_indices, test_indices)], cpu)

if config["predict_uncertainty"]:
    D2["test_test"]["K"] = jax.device_put(K[np.ix_(test_indices, test_indices)], cpu)

for i in range(len(train_indices_splits)):
    D2["train"][i]["K"] = jax.device_put(
        K[np.ix_(train_indices_splits[i], train_indices_splits[i])], cpu
    )

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

pprint(params)
# Make prediction on the test (or validation) set
Y_inference = Y[train_indices, :]
Y_predict, var = predict(Y_inference=Y_inference, D2=D2, params=params, config=config)
Y_predict = Y_scaler.inverse_transform(Y_predict)

# Print some example predictions and calculate the error
print("[Y_predict, Y_test, Abs Diff]")
print(
    np.hstack(
        [Y_predict[:5, :], Y_test[:5, :], np.abs(Y_predict[:5, :] - Y_test[:5, :])]
    )
)
MAE_mean = gpr.get_MAE(np.median(Y[all_train_indices, :]), Y_test)
MAE_barrier = gpr.get_MAE(Y_predict, Y_test)

# Sanity check
assert (
    abs(mean_absolute_error(Y_test, Y_predict) - MAE_barrier) < 1e-6
), f"{mean_absolute_error(Y_test, Y_predict)}, {MAE_barrier}"

print(f"MAE barrier predictions: {MAE_barrier:.2f}")

# Plot scatter plot of predictions vs true values and save
gpr.plot_MAE(
    color="#00509d", Y_test=Y_test, Y_predict=Y_predict, MAE=MAE_barrier, config=config
)

# Create dataframe with test predictions and perform varies sanity checks
assert np.allclose(df.index.to_numpy(), np.arange(df.shape[0]))
df_test = df.loc[test_indices].copy()
assert np.unique(df_test.split.to_numpy())[0] == "test"
assert np.sum(np.abs(df_test.E_barrier.to_numpy() - Y_test.flatten())) < 1e-6
df_test["E_barrier_predict"] = Y_predict.flatten()
assert (
    mean_absolute_error(
        df_test["E_barrier"].to_numpy(), df_test["E_barrier_predict"].to_numpy()
    )
    - MAE_barrier
) < 1e-6

df_test.set_index("hash_direction").to_pickle(config["save_df_test"])

# Print MAE for different cut-offs for trajectory systems
df_test_traj = df_test[df_test.origin == "traj"].copy()
print("Traj test scores:")
print(
    f"ALL traj: {mean_absolute_error(df_test_traj.E_barrier, df_test_traj.E_barrier_predict):.2f} kcal/mol"
)
print(
    f"3A  traj: {mean_absolute_error(df_test_traj[df_test_traj.d <= 3].E_barrier, df_test_traj[df_test_traj.d <= 3].E_barrier_predict):.2f} kcal/mol"
)
print(
    f"2A  traj: {mean_absolute_error(df_test_traj[df_test_traj.d <= 2].E_barrier, df_test_traj[df_test_traj.d <= 2].E_barrier_predict):.2f} kcal/mol"
)

print("ROOT MEAN SQUARE ERROR")
print("Traj test scores:")
print(
    f"ALL traj: {root_mean_squared_error(df_test_traj.E_barrier, df_test_traj.E_barrier_predict):.2f} kcal/mol"
)
print(
    f"3A  traj: {root_mean_squared_error(df_test_traj[df_test_traj.d <= 3].E_barrier, df_test_traj[df_test_traj.d <= 3].E_barrier_predict):.2f} kcal/mol"
)
print(
    f"2A  traj: {root_mean_squared_error(df_test_traj[df_test_traj.d <= 2].E_barrier, df_test_traj[df_test_traj.d <= 2].E_barrier_predict):.2f} kcal/mol"
)
