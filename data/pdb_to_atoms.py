"""Script used to turn pdb files in data folder to single pandas dataframe"""

from pathlib import Path
import numpy as np
import pandas as pd
from ase.io import proteindatabank
import time
from helper import transition_to_hash, clean_atoms_object, init_logger

# !!! Important !!!
#
# If data was downloaded please rename: dataset_synth -> synth, dataset_traj -> traj
#
# Before running please make sure that data is in folder "pdb" in same directory as this script
# with following structure and exact same naming
#
#   synth
#       test
#           {PDB and npz files ...}
#       train
#           {PDB and npz files ...}
#   traj
#       test
#           {PDB and npz files ...}
#       train
#           {PDB and npz files ...}
#

extended_duplicate_check = False
log = init_logger(level="debug")

# Exceptions
# Remove since energies are far lower than the rest, suggesting that there was a problem with the calculation
exceptions = {"8726771445601_8726777879769"}

seed = 0  # for shuffling later
root = Path("pdb/").resolve()
metafile_path = root / "metadata.csv"
df_meta_check = pd.read_csv(metafile_path, low_memory=False)

# Find all metafiles contained in the data folder
meta_files = set([str(x.stem) for x in root.rglob("*.npz")])
meta_files_synth = set([str(x.stem) for x in (root / "synth").rglob("*.npz")])
meta_files_traj = set([str(x.stem) for x in (root / "traj").rglob("*.npz")])
assert len(meta_files_synth) + len(meta_files_traj) == len(meta_files)

log.info(f"Number of contained METAFILES: {len(meta_files)}")
log.info(f"Number of contained SYNTH METAFILES: {len(meta_files_synth)}")
log.info(f"Number of contained TRAJ METAFILES: {len(meta_files_traj)}")

# Get a list of PDB files (forward and backward)
pdb_files = set([str(x.relative_to(root)) for x in root.rglob("*.pdb")])
pdb_files_forward = set(
    [str(x.relative_to(root)).split("_1.pdb")[0] for x in root.rglob("*_1.pdb")]
)
pdb_files_backward = set(
    [str(x.relative_to(root)).split("_2.pdb")[0] for x in root.rglob("*_2.pdb")]
)
assert pdb_files_forward == pdb_files_backward and len(pdb_files) == len(
    pdb_files_forward
) + len(pdb_files_backward)
log.info(
    f"Number of contained PDB files: {len(pdb_files)}, Number of barriers: {len(pdb_files_forward)}"
)

# All found PDB files should be registered in the metadata.csv file
log.info(f"Number of transitions in METAFILE TABLE: {df_meta_check.shape[0]}")
metafile_hashes = set(
    [f"{row['hash_u1']}_{row['hash_u2']}" for _, row in df_meta_check.iterrows()]
)
assert df_meta_check.shape[0] / 2 == len(metafile_hashes)
log.info(f"Number of barriers in METAFILE TABLE: {len(metafile_hashes)}")
log.info(
    f"Number of transitions in METAFILES but NOT in METAFILE TABLE: {len(meta_files.difference(metafile_hashes))}"
)
log.info(
    f"Number of transitions in METAFILE TABLE but NOT in METAFILES: {len(metafile_hashes.difference(meta_files))}"
)
assert metafile_hashes.issubset(meta_files)

# Take only transitions which are in the metafile table
meta_hashes = metafile_hashes
# Remove exceptions
meta_hashes = meta_hashes.difference(exceptions)

transition_i = 0
log.info(f"Looking into {root}")
tic_paths = time.time()
n_energies = 11  # number of linear steps
e_names = [f"e_{i:02}" for i in range(n_energies)]  # names of native data in metadata
expected_dir_order = (
    x for x in ["synth", "test", "train", "traj", "test", "train"]
)  # Ordering as a double check

all_atoms = []

dirs_synth_or_traj = sorted([element for element in root.iterdir() if element.is_dir()])

for dir_synth_or_traj in dirs_synth_or_traj:  # synth / traj
    assert dir_synth_or_traj.stem == (
        x := next(expected_dir_order)
    ), f"Not expected dir name - got: {dir_synth_or_traj.stem}, expected: {x}"
    if dir_synth_or_traj.is_dir():

        log.info(
            f"> {dir_synth_or_traj.relative_to(root)}                                                "
        )
        dirs_test_or_train = sorted(
            [element for element in dir_synth_or_traj.iterdir() if element.is_dir()]
        )

        for dir_test_or_train in dirs_test_or_train:  # test / train

            assert dir_test_or_train.stem == (
                x := next(expected_dir_order)
            ), f"Not expected dir name - got: {dir_synth_or_traj.stem}, expected: {x}"

            # All the numpy metadata files found (does not contain structures)
            npz_files = sorted(dir_test_or_train.glob("*.npz"))

            log.info(
                f"-> {dir_test_or_train.relative_to(root)} | Transitions found: {len(npz_files)}"
            )
            for npz_file in npz_files:

                if (
                    npz_file.stem not in meta_hashes
                ):  # Check that the hash is in the metadat.csv file
                    log.info(
                        f"{npz_file} not in the METAFILE TABLE or in Exceptions, so skip"
                    )
                    assert npz_file.stem in meta_files and (
                        npz_file.stem not in metafile_hashes
                        or npz_file.stem in exceptions
                    )
                    continue

                # PDB files associated with the transition
                pdb_files = [
                    npz_file.parent / (npz_file.stem + "_1.pdb"),
                    npz_file.parent / (npz_file.stem + "_2.pdb"),
                ]

                # PDB to ASE Atoms object
                start_atoms = proteindatabank.read_proteindatabank(pdb_files[0])
                end_atoms = proteindatabank.read_proteindatabank(pdb_files[1])

                # Remove some default information from the automatically generated ASE atoms
                clean_atoms_object(start_atoms)
                clean_atoms_object(end_atoms)

                atoms = start_atoms.copy()

                # Change the chemical symbol of the hydrogen atom to X for easier identification
                chemical_symbols = atoms.get_chemical_symbols()
                chemical_symbols[0] = "X"
                atoms.set_chemical_symbols(chemical_symbols)

                # load the metadata associated with the transition
                meta_info_i = np.load(str(npz_file), allow_pickle=True)["arr_0"].item()

                # Filter out both forward and backward transitions in metadata.csv
                df_meta_check_i = df_meta_check[
                    (df_meta_check.hash_u1 == meta_info_i["hash_u1"])
                    * (df_meta_check.hash_u2 == meta_info_i["hash_u2"])
                ]

                # Get the associated energies with each step of the linear transition
                Es = np.array([meta_info_i.get(sEi, np.nan) for sEi in e_names])

                # Create a unique hash from the transition using the start and end geometry as well as the used energies
                # This allows for easier identification of duplicates.
                transition_hash = transition_to_hash(
                    start_atoms=atoms, end_atoms=end_atoms, energies=Es
                )

                # Get the atom positions as well as the initial and end position of the hydrogen atom
                positions = atoms.get_positions()
                r_H_start = positions[0, :].copy()
                r_H_end = end_atoms.get_positions()[0, :].copy()

                # Interpolate the hydrogen atom position at each step
                ts = np.linspace(0, 1, 11)
                r_H = r_H_start + ts[..., None] * (r_H_end - r_H_start)
                r_H[np.isnan(Es), :] = (
                    np.nan
                )  # only have positions where an energy is also available

                # Set the hydrogen atom position to NaN to make it clear that it moves
                positions[0, :] = np.nan
                atoms.set_positions(positions)

                n_atoms = len(atoms)

                # There are two transition distances: One supplied in the metafile and one calculated directly from
                # the geometries. The empirically calculated one is more reliable and should be used.
                d_meta = meta_info_i["translation"]
                d_empi = np.linalg.norm(r_H_start - r_H_end)

                # Energy barrier defined as: E_max - E_init
                # Forward: E_max - E0
                # Backward: E_max - E10
                E_forward = np.nanmax(Es) - Es[0]
                E_backward = np.nanmax(Es) - Es[-1]

                # Some of the structures where optimized. If this is the case, then include this information
                if (
                    "e_s_opt" in meta_info_i
                    and "e_ts_opt" in meta_info_i
                    and "e_e_opt" in meta_info_i
                ):

                    # E_opt = E_optimized_transition_state - E_optimized_{start/end}
                    E_forward_opt = meta_info_i["e_ts_opt"] - meta_info_i["e_s_opt"]
                    E_backward_opt = meta_info_i["e_ts_opt"] - meta_info_i["e_e_opt"]
                    opt = True

                else:
                    # No optimized energies available
                    E_forward_opt = np.nan
                    E_backward_opt = np.nan
                    opt = False

                # Split: The predefined spilt of either belonging to the "test" or "train" set
                split = dir_test_or_train.stem

                # Origin: Either a system extracted from a trajectory ("traj") setting or a synthetic system ("synth")
                origin = dir_synth_or_traj.stem

                # The index of the metadata.csv file
                metafile_id = list(df_meta_check_i.index)

                # Many sanity checks

                # Check that the PDB files have the expected structure {hash_u1}_{hash_u2}_{1/2}
                # Forward
                assert (
                    (pdb_name := pdb_files[0].stem.split("_"))[0]
                    == str(meta_info_i["hash_u1"])
                    and pdb_name[1] == str(meta_info_i["hash_u2"])
                    and pdb_name[2] == "1"
                )
                # Backward
                assert (
                    (pdb_name := pdb_files[1].stem.split("_"))[0]
                    == str(meta_info_i["hash_u1"])
                    and pdb_name[1] == str(meta_info_i["hash_u2"])
                    and pdb_name[2] == "2"
                )

                # Check that atom positions are the same, except the hydrogen atom involved in the reaction
                assert np.allclose(
                    start_atoms.get_positions()[1:], end_atoms.get_positions()[1:]
                )

                # Hydrogen atom should not(!) be at the same position
                assert not np.allclose(
                    start_atoms.get_positions()[0], end_atoms.get_positions()[0]
                )

                # No periodic boundary condition
                assert np.array_equal(
                    atoms.get_pbc(), [False, False, False]
                ), f"PBC not switched off: {atoms.get_pbc()}"

                # Should be exactly 2 rows in metadata.csv associated with reaction (forward & backward)
                assert df_meta_check_i.shape[0] == 2

                # Expect 11 energies (one for each step). Some energies can be NaN
                assert len(Es) == 11

                # Sanity check the saved energies against those in the metadata.csv file
                assert (
                    np.isclose(Es[0], df_meta_check_i.iloc[0]["e_00"])
                    and np.isclose(Es[-1], df_meta_check_i.iloc[0]["e_10"])
                    and np.isclose(Es[-1], df_meta_check_i.iloc[1]["e_00"])
                    and np.isclose(Es[0], df_meta_check_i.iloc[1]["e_10"])
                    and np.isclose(Es[5], df_meta_check_i.iloc[0]["e_05"])
                    and np.isclose(Es[5], df_meta_check_i.iloc[1]["e_05"])
                )

                # Check hydrogen positions
                assert (
                    np.allclose(r_H[0], r_H_start)
                    and np.allclose(r_H[-1], r_H_end)
                    and np.allclose(r_H[0] + (r_H_end - r_H_start) * 0.5, r_H[5])
                )

                # atoms structure should have no position for the hydrogen atom
                assert np.all(np.isnan(atoms.get_positions()[0]))

                # check the transition distance
                assert np.isclose(d_meta, meta_info_i["translation"]) and np.isclose(
                    d_empi, np.linalg.norm(r_H[-1] - r_H[0])
                )
                assert np.isclose(
                    d_meta, df_meta_check.loc[metafile_id[0]]["translation"]
                ) and np.isclose(
                    d_meta, df_meta_check.loc[metafile_id[1]]["translation"]
                )

                # check the barriers against those in metadata.csv
                assert np.isclose(E_forward, df_meta_check_i.iloc[0]["Ea"])
                assert np.isclose(E_backward, df_meta_check_i.iloc[1]["Ea"])
                assert np.isclose(
                    E_forward,
                    meta_info_i[meta_info_i["e_max_key"]] - meta_info_i["e_00"],
                )
                assert np.isclose(
                    E_backward,
                    meta_info_i[meta_info_i["e_max_key"]] - meta_info_i["e_10"],
                )
                assert np.isclose(
                    df_meta_check_i.iloc[0]["Ea"],
                    df_meta_check_i.iloc[0]["e_max"] - df_meta_check_i.iloc[0]["e_00"],
                )
                assert np.isclose(
                    df_meta_check_i.iloc[1]["Ea"],
                    df_meta_check_i.iloc[0]["e_max"] - df_meta_check_i.iloc[0]["e_10"],
                )
                assert np.isclose(meta_info_i["e_max"], np.nanmax(Es))

                # check that the split and origin are as expected
                assert split in ["train", "test"] and origin in ["traj", "synth"]

                # as many atoms as there are atom positions
                assert n_atoms == len(positions)

                # trajectory data will have a "frame" value in the matadata.csv file
                if origin == "traj":
                    assert np.all(~np.isnan(df_meta_check_i["frame"].to_numpy()))
                else:
                    assert np.all(np.isnan(df_meta_check_i["frame"].to_numpy()))

                # hydrogen atom involved in transition should have X as chemical element
                assert atoms.get_chemical_symbols()[0] == "X"

                # If optimized energies are available then compare those with the metadata file
                if np.isnan(E_forward_opt) or np.isnan(E_backward_opt):
                    assert opt is False
                    assert np.isnan(df_meta_check_i.iloc[0]["Ea_opt"]) or np.isnan(
                        df_meta_check_i.iloc[1]["Ea_opt"]
                    )
                    assert (
                        ("e_s_opt" not in meta_info_i)
                        or ("e_ts_opt" not in meta_info_i)
                        or ("e_e_opt" not in meta_info_i)
                    )
                else:
                    assert opt is True
                    assert np.isclose(
                        df_meta_check_i.iloc[0]["Ea_opt"],
                        df_meta_check_i.iloc[0]["e_ts_opt"]
                        - df_meta_check_i.iloc[0]["e_s_opt"],
                    )
                    assert np.isclose(df_meta_check_i.iloc[0]["Ea_opt"], E_forward_opt)

                    assert np.isclose(
                        df_meta_check_i.iloc[1]["Ea_opt"],
                        df_meta_check_i.iloc[1]["e_ts_opt"]
                        - df_meta_check_i.iloc[1]["e_s_opt"],
                    )
                    assert np.isclose(df_meta_check_i.iloc[1]["Ea_opt"], E_backward_opt)

                    assert np.isclose(
                        df_meta_check_i.iloc[0]["e_s_opt"],
                        df_meta_check_i.iloc[1]["e_e_opt"],
                    )
                    assert np.isclose(
                        df_meta_check_i.iloc[0]["e_e_opt"],
                        df_meta_check_i.iloc[1]["e_s_opt"],
                    )
                    assert np.isclose(
                        df_meta_check_i.iloc[0]["e_ts_opt"],
                        df_meta_check_i.iloc[1]["e_ts_opt"],
                    )

                # Finally add the transition information to a list which will be converted to a dataframe later
                all_atoms.append(
                    [
                        atoms,
                        r_H,
                        Es,
                        E_forward,
                        E_backward,
                        opt,
                        E_forward_opt,
                        E_backward_opt,
                        d_empi,
                        origin,
                        split,
                        n_atoms,
                        npz_file.stem,
                        metafile_id,
                        transition_hash,
                    ]
                )

                transition_i += 1
                print(
                    f"{100 * (transition_i) / np.ceil(len(meta_hashes)) :5.2f}% of transitions done",
                    end="\r",
                    flush=True,
                )

# Create dataframe
log.info("Creating atoms df")
df = pd.DataFrame(
    all_atoms,
    columns=[
        "atoms",
        "r_H",
        "E",
        "E_forward",
        "E_backward",
        "opt",
        "E_forward_opt",
        "E_backward_opt",
        "d",
        "origin",
        "split",
        "n_atoms",
        "meta_hash",
        "metafile_id",
        "transition_hash",
    ],
)
df = df.astype(
    {
        "transition_hash": "string",
        "meta_hash": "string",
        "origin": "string",
        "split": "string",
    }
)

# Shuffle data to avoid any implicit ordering
log.debug(f"Shuffle with seed {seed}")
df = df.sample(frac=1.0, random_state=seed, replace=False, ignore_index=True)

# Remove duplicate reactions
# For some reason there will be a small minority of duplicate values ~50. Remove them based on hash
log.info("Remove duplicate transitions")
log.debug(f"Number of all transition hashes: {df['transition_hash'].to_numpy().size}")
unique_transition_hashes, counts = np.unique(
    df["transition_hash"].to_numpy(), return_counts=True
)
log.debug(f"Number of unique transition hashes: {unique_transition_hashes.size}")

duplicate_transition_hashes = unique_transition_hashes[counts > 1]
log.debug(f"Number of duplicate transition hashes: {duplicate_transition_hashes.size}")

duplicated_select = df["transition_hash"].duplicated(keep="first")
assert set(duplicate_transition_hashes) == set(df[duplicated_select]["transition_hash"])
log.debug(
    f"Duplicate transitions: {df[duplicated_select][['meta_hash', 'metafile_id']]}"
)
df = df[~duplicated_select].copy().reset_index(drop=True)

_, counts = np.unique(df["meta_hash"].to_numpy(), return_counts=True)
assert np.all(counts == 1)

# Explicit duplicate sanity check. Compare the positions of each structure with each other structure.
# If they are the same then make sure that the hydrogen reaction positions and the associated energies are different
if extended_duplicate_check:
    log.info("Sanity check that there are no accidental duplicates. Takes a while...")
    count_check = 0
    for i, atoms_a in enumerate(df["atoms"].to_numpy()):
        for j, atoms_b in enumerate(df["atoms"].to_numpy()):
            if j <= i:
                continue
            positions_a = atoms_a.get_positions()[1:, :]
            positions_b = atoms_b.get_positions()[1:, :]
            if positions_a.shape[0] == positions_b.shape[0]:
                if np.allclose(positions_a, positions_b):
                    assert (
                        not np.allclose(
                            df.iloc[i]["r_H"], df.iloc[j]["r_H"], equal_nan=True
                        )
                    ) and (
                        not np.allclose(
                            df.iloc[i]["E"],
                            df.iloc[j]["E"],
                            equal_nan=True,
                            rtol=0,
                            atol=1e-2,
                        )
                    ), f"{i} {j}"
            count_check += 1
            print(
                f"{100 * count_check / ((df.shape[0] ** 2 - df.shape[0]) / 2) :5.2f}% of transitions done",
                end="\r",
                flush=True,
            )
else:
    log.info("Skipped expanded duplicate check as requested")

log.info(
    f"Number of final transitions (1 transition = forward + backward): {df.shape[0]}"
)

# Compact version has one row for both directions
log.info("Saving compact dataset")
df.to_pickle("atoms.pkl")
