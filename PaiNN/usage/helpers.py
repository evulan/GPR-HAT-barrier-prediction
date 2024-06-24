"""Helper functions for PaiNN training"""

import numpy as np
import pandas as pd
from pathlib import Path


def MAE(*, true, predictions):
    """Return MAE"""
    assert (
        np.array(true).ndim == 1
        and np.array(predictions).ndim == 1
        and len(true) == len(predictions)
    )
    return np.mean(np.abs(np.array(true) - np.array(predictions)))


def filter_metafiles(meta_files, remove_opt=False):
    """Filter out the meta files which are not part of the atoms dataframe for comparison purposes"""
    CWD = Path(__file__).parent
    atoms = pd.read_pickle(
        CWD.parents[1] / "data" / "atoms.pkl"
    ).copy()  # Reference transitions
    atoms_stems = atoms.meta_hash.to_numpy()
    meta_file_stems = np.array([Path(meta_file).stem for meta_file in meta_files])

    # Check which transitions are in the reference dataframe
    contained = np.isin(meta_file_stems, atoms_stems)
    print(
        f"Number of meta files not in Atoms dataframe: {sum(contained == False)} from {contained.size}"
    )

    # Whether to remove optimized transitions
    if remove_opt:
        atoms_stems_opt = atoms[atoms.opt].meta_hash.to_numpy()

        meta_files_stems_contained = np.setdiff1d(meta_file_stems, atoms_stems_opt)
        not_opt = ~np.isin(meta_file_stems, atoms_stems_opt)
        contained = contained * not_opt

        print(
            f"Number of opt files to be removed: {len(meta_file_stems)-len(meta_files_stems_contained)}"
        )
        assert (len(meta_file_stems) - len(meta_files_stems_contained)) == sum(
            not_opt == False
        )

    meta_files_stems_contained = meta_file_stems[contained]

    # Return a list of meta files which can be used and their respective transition hashes
    transition_hashes_selected = (
        atoms.copy()
        .set_index("meta_hash")
        .loc[meta_files_stems_contained]["transition_hash"]
        .to_numpy()
        .copy()
    )
    meta_files_selected = np.array(meta_files)[contained].tolist()
    return meta_files_selected, transition_hashes_selected


def meta_d_to_df_d(meta_d, errors, metas_masked=None):
    """Create a dataframe from PaiNN prediction errors"""

    # Transition stems to be used
    if metas_masked is None:
        meta_d_stems = [transition["meta_path"].stem for transition in meta_d]
    else:
        meta_d_stems = [transition.stem for transition in metas_masked]

    # Transition directions
    meta_d_directions = [transition["direction"] for transition in meta_d]

    # Energy barriers
    meta_d_barriers = np.array(
        [
            (
                transition["e_max"] - transition["e_00"]
                if transition["direction"] == 1
                else transition["e_max"] - transition["e_10"]
            )
            for transition in meta_d
        ]
    )

    CWD = Path(__file__).parent
    atoms = pd.read_pickle(CWD.parents[1] / "data" / "atoms.pkl").copy()

    # Only select transitions under consideration
    atoms_selected = atoms.set_index("meta_hash").loc[meta_d_stems]

    # Sanity check: energy barriers are assumed
    meta_lookup_selected_energies = np.array(
        [
            (
                atoms_selected.iloc[i].E_forward
                if direction == 1
                else atoms_selected.iloc[i].E_backward
            )
            for i, direction in enumerate(meta_d_directions)
        ]
    )
    assert np.allclose(meta_lookup_selected_energies, meta_d_barriers)

    # Prediction errors and actual energy prediction
    errors = np.array(errors, dtype=np.float64)
    GNN_E_barrier_predicted = meta_lookup_selected_energies + errors

    # Create a dataframe with test data that has true and predicted energy barriers
    df_d = {
        "transition_hash": atoms_selected.transition_hash.to_numpy(),
        "path": meta_d_stems,
        "direction": [
            "forward" if direction == 1 else "backward"
            for direction in meta_d_directions
        ],
        "E_barrier": meta_lookup_selected_energies,
        "GNN_errors": errors,
        "GNN_E_barrier_predicted": GNN_E_barrier_predicted,
    }
    df_d = pd.DataFrame.from_dict(df_d)
    print(df_d)
    # Return the test prediction dataframe
    return df_d
