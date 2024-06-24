import numpy as np
import pandas as pd
from mpi4py import MPI
from graph_helpers import all_atoms_to_networkx
from datetime import datetime


def mpi_networks(df, rank, comm, size, config):
    """
    Create networkx graphs from atoms objects in distributed and parallel fashion

    :param df: dataframe with atoms objects
    :param rank: the rank of the current SLURM process
    :param comm: common MPI communicator
    :param size: number of total SLURM ranks
    :param config: config file for graph creation
    :return: dataframe with networkx graphs
    """
    n_atoms = df.shape[0]

    # Each atoms object has an index. We split up the indices into almost equally-sized batches for each rank
    indices_split = np.array_split(np.arange(n_atoms), size)

    # The rank variable is different for each SLURM process starting at 0 to size-1
    # Select the index-batch for which this rank is responsible
    my_indices = indices_split[rank]
    # Select the corresponding dataframe rows
    my_df = df.iloc[my_indices]

    # Each rank process creates graphs from their unique slice
    my_collect_networks = all_atoms_to_networkx(
        my_df,
        r_cut=config["r_cut"],
        neigh_max=config["neigh_max"],
        HR_radius=config["HR_radius"],
    )

    # All ranks except for the root rank (0) send their graphs to the root rank
    if rank != 0:
        comm.send(my_collect_networks, dest=0, tag=rank)
        collect_networks = None
    # The root rank collects all graphs sent by the other ranks
    else:
        collect_networks = my_collect_networks
        # For each worker rank wait until it has sent their graphs and add it to the collection
        for worker_i in range(1, size):
            worker_df = comm.recv(source=worker_i, tag=worker_i)
            collect_networks = pd.concat([collect_networks, worker_df])
        collect_networks.reset_index(drop=True, inplace=True)
        # Sanity check that the unique reaction oder is the same as in the original dataframe
        assert np.array_equal(
            np.unique(df.transition_hash.to_numpy()),
            np.unique(collect_networks.transition_hash.to_numpy()),
        )

    collect_networks = comm.bcast(collect_networks, root=0)

    return collect_networks


def get_mpi_R(graphs, n_workers, n_splits, rank, comm, kernel):
    """
    Create almost triangular R kernel matrix block-wise with each rank process
    :param graphs: Networkx graphs based on atomic structures
    :param n_workers: number of SLURM ranks
    :param n_splits: number of blocks in one dimension, i.e. matrix will be split into n_splits**2 blocks
    :param rank: current rank of SLURM process executing function
    :param comm: MPI communicator
    :param kernel: MGK kernel from GraphDot
    :return: Block-wise triangular R kernel matrix
    """
    graphs = np.array(graphs)
    n = len(graphs)
    indices = np.arange(n)

    # Batch of indices for each block in one dimension
    block_graph_indices = np.array_split(indices, n_splits).copy()
    # Create nested list with each element containing the indices of graphs to use in x and y direction
    # Since the kernel matrix is symmetrical, only use blocks which are in the upper triangle of the matrix.
    block_indices = [
        [block_graph_indices[i], block_graph_indices[j]]
        for i in range(n_splits)
        for j in range(i, n_splits)
    ]
    # Get block shapes
    block_shapes = [[len(indices[0]), len(indices[1])] for indices in block_indices]
    # Create a list where each element contains the blocks for which a worker is responsible
    enumeration_of_blocks = np.arange(len(block_indices))
    assigned_blocks_to_workers = np.array_split(enumeration_of_blocks, n_workers)

    # Get the blocks which are assigned to the current worker/rank
    this_worker_blocks = assigned_blocks_to_workers[rank]
    this_worker_block_indices = [block_indices[i] for i in this_worker_blocks]

    # Calculate the kernel values for the rank blocks
    this_worker_result = worker_job(this_worker_block_indices, graphs, kernel, rank)

    # If not root rank then send the results to root
    if rank != 0:
        for i, worker_block_index in enumerate(this_worker_blocks):
            # Important! MPI assumes C memory order, but numpy array might not be, therefore use ascontiguousarray
            comm.Send(
                [np.ascontiguousarray(this_worker_result[i]), MPI.DOUBLE],
                dest=0,
                tag=worker_block_index,
            )
        return None  # and then exit
    # If root rank then collect results
    else:
        print("Collect all R blocks")
        R = np.zeros((n, n))
        # For each worker collect each of their calculated blocks
        for worker_i in range(0, n_workers):
            worker_block_indices = assigned_blocks_to_workers[worker_i]
            for i, worker_block_index in enumerate(worker_block_indices):
                block_shape = block_shapes[worker_block_index]
                block_graph_indices = block_indices[worker_block_index]

                # root is by assignment rank 0, so already has its result
                if worker_i == 0:
                    worker_block_result = this_worker_result[i]
                # for all other ranks wait to receive their results
                else:
                    worker_block_result = np.zeros(
                        block_shape, dtype=np.double
                    )  # empty array to be filled
                    comm.Recv(
                        [worker_block_result, MPI.DOUBLE],
                        source=worker_i,
                        tag=worker_block_index,
                    )
                # Write the worker block result to the corresponding block in the final kernel matrix
                R[np.ix_(block_graph_indices[0], block_graph_indices[1])] = (
                    worker_block_result.copy()
                )
        return R


def worker_job(this_worker_block_indices, graphs, kernel, rank):
    """
    Calculates the kernel matrix blocks assigned to the worker

    :param this_worker_block_indices: the graph indices of the block
    :param graphs: all graphs
    :param kernel: GraphDot kernel function
    :param rank: worker rank
    :return: list of calculated blocks
    """
    result = []
    for splits in this_worker_block_indices:
        splits_i, splits_j = splits[0], splits[1]
        graphs_i, graphs_j = graphs[splits_i], graphs[splits_j]
        print(
            f"rank {rank} | Starting splits ({len(splits_i)},{len(splits_j)}) at {datetime.now().strftime('%H:%M:%S')}",
            flush=True,
        )
        result.append(kernel(graphs_i, graphs_j, lmin=0))
    return result
