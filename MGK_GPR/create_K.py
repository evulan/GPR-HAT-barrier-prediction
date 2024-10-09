import time
from pprint import pprint, pformat
from datetime import datetime
import numpy as np
import pandas as pd
from graphdot import Graph
from mpi4py import MPI
from mpi_helpers import mpi_networks, get_mpi_R
from pathlib import Path
import sys

sys.path.append(str(Path.cwd().parent.resolve()))

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Sometimes SLURM nodes fail to load graphdot. This notifies the user which node is problematic
try:
    from graphdot.kernel.marginalized import MarginalizedGraphKernel
    from graphdot.kernel.basekernel import (
        TensorProduct,
        SquareExponential,
        KroneckerDelta,
        Composite,
    )
    from graphdot.kernel.marginalized.starting_probability import Adhoc
except Exception as e:
    print(rank, MPI.Get_processor_name(), e)
    raise Exception

ignore_side_feature = False
# ignore_side_feature = True

# The kernel to use for nodes/atoms

if not ignore_side_feature:
    knode = TensorProduct(
        atomic_number=KroneckerDelta(0.8),
        special_type=KroneckerDelta(0.2),
        alpha=SquareExponential(1.0),
    )
else:
    knode = TensorProduct(
        atomic_number=KroneckerDelta(0.8),
        special_type=KroneckerDelta(0.2),
    )

# The kernel to use for edges
kedge = TensorProduct(
    length=SquareExponential(1.0),
)

config = dict(
    r_cut=3,  # maximum radius of neighborhood for edge consideration for each atom
    neigh_max=5,  # maximum number of connected edges allowed per atom
    HR_radius=5,  # maximum distance from start, middle or end positions allowed for atom still to be considered
    knode=str(knode),  # node kernel
    kedge=str(kedge),  # edge kernel
    p_q=0.05,  # end probability
    n_split=10,  # number of blocks to use for MPI in one dimension, i.e. n_split * n_split = total num of blocks
    K_save=f"K.npy",  # File where to save kernel matrix
    df_save=f"df_MGK.pkl",  # modified atoms dataframe used for the creation
    collect_networks_save="collect_networks_diff.pkl",  # save file name of the graphs that were created from the atoms
    workers=size,  # number of workers
    n_max=None,  # maximum number of transitions to use for testing. Use None to use all
    normalise_K=True,  # Whether to normalize K
)

# the random walk starting probability should be equal for start, niddle and end hyrogen atom and 0 otherwise
p_start = Adhoc(
    f=lambda nodes: np.where(
        (nodes.special_type == 1)
        or (nodes.special_type == 2)
        or (nodes.special_type == 3),
        0.3333,
        0.0,
    ),
    expr="n.special_type == 1 || n.special_type == 2 || n.special_type == 3? 0.3333f : 0.0f",
)

# initialize kernel function
kernel = MarginalizedGraphKernel(knode, kedge, p=p_start, q=config["p_q"])

# load atom structures
df = pd.read_pickle("../data/atoms.pkl")
df = df.iloc[: config["n_max"]]

print("Start Time:", datetime.now().strftime("%H:%M:%S"), flush=True)
tic = time.time()

# Create networkx graphs from atom structures using multiple nodes
collect_networks = mpi_networks(df, rank, comm, size, config, ignore_side_feature)

# Create GraphDot graphs from networkx graphs
graphs = [Graph.from_networkx(m, weight="w") for m in collect_networks.graph]

# Calculate the kernel matrix blockwise with MPI
print(rank, "Start Time R:", datetime.now().strftime("%H:%M:%S"))
tic_R = time.time()
R = get_mpi_R(graphs, size, config["n_split"], rank, comm, kernel)
toc_R = time.time()
toc = toc_R

# Terminate all ranks except root
if rank != 0:
    print(rank, "worker finished", flush=True)
    exit()
else:
    print(rank, f"R ({R.shape}) took {toc_R - tic_R:.2f}s")
    print(rank, f"Total {toc - tic:.2f}s", flush=True)


# Kernel matrix will not be symmetrical yet. Do this now
def symmetrise_R(R):
    u = np.triu_indices(R.shape[0], 1)
    R.T[u] = R[u]
    return R


print("Next symmetrize R.")
R = symmetrise_R(R)

print(R)
print(f"R_min: {R.min()}, R_max: {R.max()}", flush=True)
assert np.any(~np.isnan(R))

# Normalize the matrix if specified
if config["normalise_K"]:
    print("Normalize R matrix")
    d = np.clip(np.diag(R), 0, np.inf) ** 0.5
    K = R / d[..., None] / d[None, ...]
    K = np.clip(K, -1, 1.0)
    print("New R")
    print(K)
    print(f"R_min: {K.min()}, R_max: {K.max()}", flush=True)
    assert np.any(~np.isnan(K))
else:
    print("Do not normalise R matrix")
    K = R

print(f"Save R matrix as {config['K_save']}")
np.save(config["K_save"], K)

print("Save rest")
df.to_pickle(config["df_save"])
collect_networks.to_pickle(config["collect_networks_save"])

with open("creation_config.txt", "w") as f:
    f.write(f"{pformat(config, indent=4)}")
pprint(config)
