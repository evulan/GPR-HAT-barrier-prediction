"""Utility functions used for the GPR calculations throughout the project"""

import numpy as np
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import jax
from jax import numpy as jnp


def set_jax(accelerator, float64=True, gpu_requested=True):
    """Sets the default device of jax arrays (either CPU or GPU) and floating point precision"""
    CPU = jax.devices("cpu")[0]
    GPU = jax.devices("gpu")[0] if gpu_requested else None
    print("Devices: ", CPU, GPU)

    if accelerator == "cpu":
        JIT_DEFAULT_DEVICE = CPU
        jax.config.update("jax_platform_name", "cpu")
    else:
        JIT_DEFAULT_DEVICE = GPU

    jax.config.update("jax_enable_x64", float64)
    print(f"JIT_DEFAULT_DEVICE: {JIT_DEFAULT_DEVICE}")
    test_arr = jnp.ones(3)
    print("Test array:", (test_arr.device(), test_arr.dtype))

    return str(JIT_DEFAULT_DEVICE), test_arr.dtype


def get_D(*, X, exponent):
    """Create a distance matrix and cast to jax array"""
    return jnp.array(pairwise_distances(X=X, n_jobs=-1) ** exponent)


def SE_paper(params, D2):
    """Squared Exponential kernel"""
    lambda2 = params["lambda"] ** 2
    return jnp.exp(-D2 / lambda2)


def RQ(params, D2):
    """Rational Quadratic kernel"""
    a = params["alpha"] ** 2
    l = params["l"]
    return (1 + (D2 / (2 * a * (l**2)))) ** (-a)


def M12(params, D):
    """Matern 1/2 kernel"""
    l = params["l"] ** 2
    return jnp.exp(-l * D)


def M32(params, D):
    """Matern 3/2 kernel"""
    l = params["l"] ** 2
    A = jnp.sqrt(3) * l * D
    return (1 + A) * jnp.exp(-A)


def get_diagL_a(K, Y):
    """Cholesky decomposition. Part of GPR calculation to avoid naive inversion"""
    L = jax.scipy.linalg.cholesky(K, lower=True)
    alpha = jax.scipy.linalg.cho_solve((L, True), Y)
    return jnp.diagonal(L), alpha


def NLL(K, Y_train):
    """Negative Log Likelihood of K and Y_train"""
    diagL, alpha = get_diagL_a(K, Y_train)
    return (0.5 * Y_train.T @ alpha + jnp.log(diagL).sum())[0, 0]


def K_transform_general(K, params, config):
    """General correlation map: K -> s**2*(K+g)+jitter"""
    n = K.shape[0]
    return (params["general"]["sigma"] ** 2) * (
        K + (params["general"]["nugget"] ** 2) * jnp.eye(n)
    ) + config["jitter"] * jnp.eye(n)


def get_MAE(Y_predict, Y_true):
    """Return MAE"""
    return (np.abs(Y_predict.flatten() - Y_true.flatten())).mean()


@jax.jit
def predict_mean(K_star, alpha):
    """Mean predictions of GPR"""
    return K_star.T @ alpha


def plot_MAE(color="#00509d", *, Y_test, Y_predict, MAE, config):
    """Plot the predictions vs true values"""
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
        Y_test.flatten(),
        Y_predict.flatten(),
        s=20,
        marker="o",
        alpha=0.5,
        linewidths=0.5,
        color=color,
        edgecolors="black",
        label="Predictions",
    )

    ax.set_aspect("equal", "box")
    ax.set_xlabel(r"$\Delta E^{True}\:\left[\frac{kcal}{mol}\right]$")
    ax.set_ylabel(r"$\Delta E^{Predicted}\:\left[\frac{kcal}{mol}\right]$")

    # ticks = (t := ax.get_yticks())[t >= 0.0]  # Remove negative axis part
    # Check that no point is accidentally beyond tick range
    # assert (
    #     Y_test.min() >= ticks.min()
    #     and Y_test.max() <= ticks.max()
    #     and Y_predict.min() >= ticks.min()
    #     and Y_predict.max() <= ticks.max()
    # )
    # ax.set_xticks(ticks)
    # ax.set_xticklabels(ticks.astype(int), rotation=-45)
    # ax.set_yticks(ticks)
    # ax.set_yticklabels(ticks.astype(int), rotation=0)
    diag = np.linspace(
        np.max([ax.get_xlim()[0], 0.0]), ax.get_xlim()[1], 1000
    )  # Diagonal line
    ax.plot(
        diag, diag, color="black", ls="--", zorder=0, label="Perfect prediction", lw=2
    )

    ax.annotate(
        rf"MAE: ${MAE:.2f}\:\frac{{kcal}}{{mol}}$",
        (0.02, 0.95),
        xycoords="axes fraction",
        fontsize=24,
        bbox=dict(boxstyle="square,pad=0.3", alpha=0.0),
    )

    plt.savefig(
        config["plot_save"],
        dpi=400,
        bbox_inches="tight",
        metadata={str(k): str(v) for k, v in config.items()},
    )
    plt.show()


def use_empirical_d(*, use_empirical, df):
    """Choose whether to use empirically calculated transition distances from the atomic structures"""

    # d_empi is the distance calculated using ASE, d_meta is the distance included with the original dataset
    if use_empirical:
        df["d"] = df["d_empi"].copy()
        assert np.allclose(df["d"].values, df["d_empi"].values)
    else:
        df["d"] = df["d_meta"].copy()
        assert np.allclose(df["d"].values, df["d_meta"].values)
    return df


def filter(config, df, verbose=True):
    """Creates masks for the test and training cases. The masks are same length as the total number of points"""
    test_criteria = df.split == "test"
    train_criteria = df.split == "train"

    # filter test points by distance bounds
    if config["test_d_bounds"]:
        test_criteria *= (config["test_d_bounds"][0] <= df.d) * (
            df.d <= config["test_d_bounds"][1]
        )

    # filter training points by distance bounds
    if config["train_d_bounds"]:
        train_criteria *= (config["train_d_bounds"][0] <= df.d) * (
            df.d <= config["train_d_bounds"][1]
        )

    # filter test points type of data origin, i.e. synthetic or trajectory
    if config["test_origin"]:
        test_criteria *= df.origin == config["test_origin"]

    # filter training points type of data origin, i.e. synthetic or trajectory
    if config["train_origin"]:
        train_criteria *= df.origin == config["train_origin"]

    # filter test points to only include those with an optimized barrier
    if config["test_only_opt"]:
        test_criteria *= (~pd.isna(df.E_forward_opt)) * (~pd.isna(df.E_backward_opt))

    # filter training points to only include those with an optimized barrier
    if config["train_only_opt"]:
        train_criteria *= (~pd.isna(df.E_forward_opt)) * (~pd.isna(df.E_backward_opt))

    if verbose:
        print("TRAIN filtered df sample:")
        print(
            df[train_criteria].iloc[:3],
            "\nTRAIN n_points:",
            df[train_criteria].shape[0],
        )
        print(
            f"TRAIN d bounds: [{df[train_criteria].d.min()},{df[train_criteria].d.max()}]"
        )
        print(f"TRAIN Origins: {df[train_criteria].origin.unique()}")

        print("TEST filtered df sample:")
        print(
            df[test_criteria].iloc[:3], "\nTEST n_points:", df[test_criteria].shape[0]
        )
        print(
            f"TEST d bounds: [{df[test_criteria].d.min()},{df[test_criteria].d.max()}]"
        )
        print(f"TEST Origins: {df[test_criteria].origin.unique()}")

    return train_criteria, test_criteria


def prepare_df(df, config):
    """Prepare dataframe for usage in GPR, especially if a validation set instead of actual test set should be used"""
    n_train_init = (df.split == "train").sum()  # number of original training points
    n_train = np.nan

    if not config[
        "test_production"
    ]:  # if no validation set should be used, but the actual test set
        print("Sampling validation points")

        # Sample uniform randomly validation points from the training set
        df.loc[df.split == "test", "split"] = (
            "ori_test"  # actual test points are now designated ori_test
        )
        val_indx = (
            df[df.split == "train"]
            .sample(frac=1 / 9, random_state=config["seed"], replace=False)
            .index.to_numpy()
            .copy()
        )
        df.loc[val_indx, "split"] = (
            "test"  # The sampled validation points are now the "test" points
        )
    else:
        print("INFO: Using real test cases")

    # Get boolean filters based on configuration. e.g. only use points within a certain transition distance d range
    # Note: This uses now the newly defined test/train split above
    train_criteria, test_criteria = filter(config, df, verbose=False)

    # Now create a new dataframe with potentially fewer training samples
    df["split_init"] = df["split"].copy()
    df["split"] = "None"  # Remove original split designation

    df_train = df[train_criteria.to_numpy()].copy()
    df_test = df[test_criteria.to_numpy()].copy()

    # Get number of training points to sample based on configuration
    if type(config["validation"]["n_train"]) is float:
        n_train = int(n_train_init * config["validation"]["n_train"])
    elif type(config["validation"]["n_train"]) is int:
        n_train = config["validation"]["n_train"]

    # If "all_sequential" was chosen then all training points are taken in the order of appearance in the training set
    if config["validation"]["n_train"] == "all_sequential":
        n_train = train_criteria.sum()
        train_indx = df_train.index.to_numpy().copy()
    else:
        train_indx = (
            df_train.sample(n=n_train, random_state=config["seed"], replace=False)
            .index.to_numpy()
            .copy()
        )

    # Choose the original validation or true test set as defined previously
    val_indx = df_test.index.to_numpy()

    # Designate new splits
    df.loc[train_indx, "split"] = "train"
    df.loc[val_indx, "split"] = "test"

    print(
        f"Select n_train: {df[df.split == 'train'].shape[0]}, n_test: {df[df.split == 'test'].shape[0]}"
    )

    # Perform some sanity checks
    assert df[df.split == "train"].shape[0] == n_train
    # Check that training and validation are separate
    assert (
        np.intersect1d(df[df.split == "train"].index, df[df.split == "test"].index).size
        == 0
    ), "Sanity check failed. Test and train not separate"
    # Check that if we chose a validation split, it is NOT part of the original test set
    if not config["test_production"]:
        assert (
            np.intersect1d(
                df[df.split == "test"].index, df[df.split_init == "ori_test"].index
            ).size
            == 0
        ), "Sanity check failed. Actual test and validation not separate"

    return df.copy()


def set_D2_soap(
    loc,
    D2_dict,
    train_indices_splits,
    test_indices,
    transition_hashes,
    directions,
    cpu,
    name,
    exponent=2,
):
    """Load the precalculated SOAP vectors distance matrices"""
    print(f"Loading D {name}, using D exponent {exponent}!!!")

    # Load the precomputed SOAP distance matrices
    Z = np.load(loc, allow_pickle=True)

    transition_hashes_load = Z["transition_hashes"]
    direction_load = Z["direction"]
    D2_full = Z["D"]  # Distance matrices

    # Sanity checks that the SOAP distance matrix order is the same as of the data set at hand
    assert np.array_equal(
        transition_hashes_load, transition_hashes
    ), "Transition hashes NOT EQUAL!"
    assert np.array_equal(direction_load, directions), "Directions NOT EQUAL!"

    # Sanity check that we use the correct distance matrix that was computed for the correct position in the transition
    # For different type of SOAP distance matrices computed. In this project only '"/" not in i_position' is relevant
    i_position = name.split("_")[1]
    if "/" not in i_position:
        assert (
            i_position == f'{Z["config"].item()["i_position"]:.1f}'
        ), f"Not comparing same positions! Assumed: {i_position}, Loaded: {Z['config'].item()['i_position']}"
    else:
        assert (
            i_position == Z["config"].item()["i_position"]
        ), f"Not comparing same positions! Assumed: {i_position}, Loaded: {Z['config'].item()['i_position']}"

    all_train_indices = np.array(train_indices_splits).flatten()

    # We save 4 different sub-parts of the distance matrices: D(train,test), D(test,test), D(train,train) and D(train_batch_i, train_batch_i)

    D2_dict["train_test"]["soap"][name] = jax.device_put(
        D2_full[np.ix_(all_train_indices, test_indices)] ** exponent, cpu
    )
    D2_dict["test_test"]["soap"][name] = jax.device_put(
        D2_full[np.ix_(test_indices, test_indices)] ** exponent, cpu
    )
    D2_dict["train_all"]["soap"][name] = jax.device_put(
        D2_full[np.ix_(all_train_indices, all_train_indices)] ** exponent, cpu
    )

    # For each batch use only the block of the training indices of the batch
    for i in range(len(train_indices_splits)):
        D2_dict["train"][i]["soap"][name] = jax.device_put(
            D2_full[np.ix_(train_indices_splits[i], train_indices_splits[i])]
            ** exponent,
            cpu,
        )


def randomise_initial_parameters(params, seed):
    """For the SOAP methods randomise the initial kernel parameters"""
    np.random.seed(seed)
    params_random = {
        # Appeared to work better than broad uniform initialization
        "general": {
            "nugget": abs(float(np.random.normal(0.1, 0.02))),
            "sigma": float(np.random.uniform(1e-6, 10)),
        },
        "soap": {
            "s_0.0": {"lambda": float(np.random.uniform(1e-6, 10))},
            "s_5.0": {"lambda": float(np.random.uniform(1e-6, 10))},
            "s_10.0": {"lambda": float(np.random.uniform(1e-6, 10))},
        },
        "d": {"lambda": float(np.random.uniform(1e-6, 10))},
    }

    # Sanity checks that the newly initialized parameters contain the same keys
    assert (
        params.keys() == params_random.keys()
    ), f"{params.keys()}, {params_random.keys()}"
    assert (
        params["general"].keys() == params_random["general"].keys()
    ), f"{params['general'].keys()}, {params_random['general'].keys()}"
    if "soap" in params.keys():
        assert (
            params["soap"].keys() == params_random["soap"].keys()
        ), f"{params['soap'].keys()}, {params_random['soap'].keys()}"

    return params_random
