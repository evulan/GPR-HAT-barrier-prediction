import numpy as np
import pandas as pd
import networkx as nx
from ase import neighborlist, Atom


def all_atoms_to_networkx(
    df, r_cut=20, neigh_max=5, HR_radius=15, collect_pruned=False
):
    """
    Main function which creates networkx graphs from atoms dataframe

    :param df: dataframe containing atoms data
    :param r_cut: maximum radius of neighborhood under consideration
    :param neigh_max: maximum number of neighbors of a given atom
    :param HR_radius: maximum allowed radius of neighborhood around the start, middle or end hydrogen atom
    :param collect_pruned: Whether to also return the atoms before and after the deletion of atoms
    :return: dataframe containing networkx graphs
    """
    collector = []
    if collect_pruned:
        pruned_atoms_collector = []
    indices = df.index
    for i, index in enumerate(indices):
        if i % 50 == 0:
            print(f"At i = {i} and index = {index}")

        df_i = df.loc[index, :]
        atoms = df_i.atoms.copy()
        positions = atoms.get_positions().copy()
        ChSy = np.array(atoms.get_chemical_symbols())
        ChSy[0] = "H"
        atoms.set_chemical_symbols(ChSy)

        Es = df_i.E.copy()
        d = df_i["d"]
        transition_hash = df_i.transition_hash
        split = df_i.split
        origin = df_i.origin

        r_H = df_i.r_H.copy()
        r_H_start = r_H[0]
        r_H_end = r_H[-1]

        positions[0, :] = r_H_start.copy()
        atoms.set_positions(positions)

        r_H_mid = r_H_start + 0.5 * (r_H_end - r_H_start)
        H_mid = Atom("H", position=r_H_mid.copy())
        atoms.append(H_mid)

        H_end = Atom("H", position=r_H_end.copy())
        atoms.append(H_end)

        E_barriers = [np.nanmax(Es) - Es[0], np.nanmax(Es) - Es[-1]]

        for i_direction, direction in enumerate(["forward", "backward"]):

            hash_direction = f"{transition_hash}_{direction}"
            E_barrier = E_barriers[i_direction]

            if direction == "backward":
                positions = atoms.get_positions().copy()
                positions[0, :] = r_H_end.copy()
                positions[-1, :] = r_H_start.copy()
                atoms.set_positions(positions)

            graph, atoms_pruned = to_networkx(
                atoms,
                r_cut=r_cut,
                neigh_max=neigh_max,
                HR_radius=HR_radius,
                collect_pruned=collect_pruned,
            )

            graph_hash = nx.weisfeiler_lehman_graph_hash(graph)

            collector.append(
                [
                    hash_direction,
                    transition_hash,
                    direction,
                    graph,
                    graph_hash,
                    E_barrier,
                    d,
                    split,
                    origin,
                ]
            )
            if collect_pruned:
                pruned_atoms_collector.append(atoms_pruned)

    collector = np.array(collector, dtype=object)
    df = pd.DataFrame(
        collector,
        columns=[
            "hash_direction",
            "transition_hash",
            "direction",
            "graph",
            "graph_weisfeiler_hash",
            "E_barrier",
            "d",
            "split",
            "origin",
        ],
    )
    df = df.astype(
        {
            "hash_direction": "string",
            "transition_hash": "string",
            "direction": "string",
            "graph": "object",
            "graph_weisfeiler_hash": "string",
            "E_barrier": "float64",
            "d": "float64",
            "origin": "string",
            "split": "string",
        }
    )
    if not collect_pruned:
        return df
    else:
        return df, pruned_atoms_collector


def to_networkx(atoms, r_cut=15, neigh_max=5, HR_radius=5, collect_pruned=False):
    """
    Creates a networkx graph from atoms object

    :param atoms: ASE Atoms object
    :param r_cut: maximum radius of neighborhood under consideration
    :param neigh_max: maximum number of neighbors of a given atom for edge creation
    :param HR_radius: maximum allowed radius of neighborhood around the start, middle or end hydrogen atom
    :param collect_pruned: Whether to also return the atoms before and after the deletion of atoms
    :return: Networkx graph
    """

    atoms_graph = nx.Graph()
    atoms = atoms.copy()
    if collect_pruned:
        atoms_before = atoms.copy()
    n_atoms = len(atoms)

    # Indices of hydrogen which undergoes transition at start, middle and end positions
    i_H_A = 0
    i_H_M = n_atoms - 2
    i_H_B = n_atoms - 1

    all_atoms_i = np.arange(n_atoms)

    # Get distances of each atom to start, middle and end positions. Select only those which are closer than HR_radius

    dist_to_H_A = atoms.get_distances(i_H_A, all_atoms_i)
    closest_H_A = all_atoms_i[dist_to_H_A <= HR_radius].copy()
    dist_to_H_B = atoms.get_distances(i_H_B, all_atoms_i)
    closest_H_B = all_atoms_i[dist_to_H_B <= HR_radius].copy()
    dist_to_H_M = atoms.get_distances(i_H_M, all_atoms_i)
    closest_H_M = all_atoms_i[dist_to_H_M <= HR_radius].copy()

    del atoms[
        [
            atom.index
            for atom in atoms
            if atom.index not in closest_H_A
            and atom.index not in closest_H_B
            and atom.index not in closest_H_M
            and atom.index != i_H_A
            and atom.index != i_H_B
            and atom.index != i_H_M
        ]
    ]

    n_atoms = len(atoms)
    atoms_indices = range(n_atoms)

    i_H_A = atoms_indices[0]
    assert i_H_A == 0
    i_H_M = atoms_indices[-2]
    assert i_H_M == n_atoms - 2
    i_H_B = atoms_indices[-1]
    assert i_H_B == n_atoms - 1

    atomic_numbers = atoms.get_atomic_numbers()

    positions = atoms.get_positions().copy()

    # Calculate the normalized vector in direction from start to end position
    transition_direction = positions[i_H_B] - positions[i_H_A]
    transition_direction = transition_direction / np.linalg.norm(transition_direction)

    # Loop through each atom and create a graph node
    for i in atoms_indices:

        # assign the hydrogen atom undergoing the transition a special type
        if i == i_H_A:
            special_type = 1
        elif i == i_H_B:
            special_type = 2
        elif i == i_H_M:
            special_type = 3
        else:
            special_type = 0

        a = atomic_numbers[i]

        # Calculate the normalized vector from atom position to the middle hydrogen atom
        dir_to_M = (
            positions[i] - positions[i_H_M] if i != i_H_M else transition_direction
        )
        dir_to_M = dir_to_M / np.linalg.norm(dir_to_M)

        # Alpha is the dot product of transition direction and vector pointing to the middle of the transition
        alpha = round(transition_direction @ dir_to_M, 6)

        attributes = dict(
            atomic_number=a,
            special_type=special_type,
            alpha=alpha,
        )
        # Add the new node with attributes
        atoms_graph.add_node(node_for_adding=i, **attributes)

    # Create a neighbourhood around each atom using r_cut beyond which atoms cannot be neighbours
    artificial_cutoffs = [r_cut] * n_atoms
    nl_artificial = neighborlist.NeighborList(
        artificial_cutoffs, self_interaction=False, bothways=True
    )
    nl_artificial.update(atoms)

    # Loop though each atom and consider its local neighbourhood for edge creation
    for n_i in range(n_atoms):
        neighs_artificial, _ = nl_artificial.get_neighbors(n_i)

        # Always make transition hydrogen atom connected at different positions
        if n_i == i_H_A:
            if i_H_B not in neighs_artificial:
                np.append(neighs_artificial, i_H_B)
            elif i_H_M not in neighs_artificial:
                np.append(neighs_artificial, i_H_M)

        # If neighbourhood empty skip
        if len(neighs_artificial) == 0:
            continue

        # Get distances to neighbourhood atoms
        dists_from_n_i = atoms.get_distances(n_i, neighs_artificial)

        # Sort the distances and only consider the neigh_max closest atoms for edge creation
        sorted_by_distance_indices = np.argsort(dists_from_n_i)
        sorted_distances = dists_from_n_i[sorted_by_distance_indices][:neigh_max]
        neighs_artificial = neighs_artificial[sorted_by_distance_indices][:neigh_max]

        # Loop though all atoms which are left in the neighbourhood
        for j in range(len(neighs_artificial)):

            n_j = neighs_artificial[j]

            # Skip if edge already exists
            if atoms_graph.has_edge(n_i, n_j):
                continue

            edge_length = sorted_distances[j]

            # Calculate weight of edge based on the typical bond length of the chemical elements using gaussian
            w = custom_atom_adjacency(
                atoms[n_i].number, atoms[n_j].number, edge_length, scale=1.0
            )

            attributes = dict(
                w=w,
                length=edge_length,
            )
            # Add the edge with attributes
            atoms_graph.add_edge(u_of_edge=n_i, v_of_edge=n_j, **attributes)

    if collect_pruned:
        atoms_after = atoms.copy()
        return atoms_graph, (atoms_before, atoms_after)
    else:
        return atoms_graph, None


def get_bond_lengths(atomic_number_i, atomic_number_j):
    """
    Returns the typical bond lengths between two atoms of different atomic numbers

    Values taken from 'Prediction of atomization energy using graph kernel and active learning'
    J. Chem. Phys. 150, 044107 (2019); https://doi.org/10.1063/1.5078640

    :param atomic_number_i: atomic number of first atom
    :param atomic_number_j: atomic number of second atom
    :return: Typical bond length [Å]
    """
    lengths = np.array(
        [
            [0.74, 1.09, 0.96, 1.01, 1.34],
            [1.09, 1.39, 1.27, 1.34, 1.82],
            [0.96, 1.27, 1.48, 1.23, 1.44],
            [1.01, 1.34, 1.23, 1.26, 1.68],
            [1.34, 1.82, 1.44, 1.68, 2.05],
        ]
    )
    atmnumbers = [1, 6, 8, 7, 16]
    df = pd.DataFrame(data=lengths, index=atmnumbers, columns=atmnumbers)
    return df.loc[atomic_number_i, atomic_number_j]


def custom_atom_adjacency(atomic_number_i, atomic_number_j, d, scale=1.0):
    """Adjacency rule which scales weight based on Gaussian of distance relative to typical bond lengths"""
    return np.exp(
        -0.5
        * (d**2)
        / (scale * get_bond_lengths(atomic_number_i, atomic_number_j) ** 2)
    )
