"""Script which creates SOAP vectors centered on specified HAT steps for all systems and saves the distance matrix"""

from dscribe.descriptors import SOAP
from ase import Atom
import pandas as pd
import time
import numpy as np
import uuid
from sklearn.metrics import pairwise_distances
from pprint import pformat
from pathlib import Path
from datetime import datetime
import sys
import logging


# Expect a command line argument at which step of the reaction should the SOAP center be used. 0->start, 5->middle, 10->end
assert len(sys.argv) == 2 and sys.argv[1].isdigit() and (0 <= int(sys.argv[1]) <= 10)
i_position = int(sys.argv[1])

cwd = Path(__file__).resolve().parent  # current working dir
hashid = str(
    uuid.uuid4()
)  # Each run has a unique automatically generated hash for later identification

config = dict(
    # The step at which the center of the features should be calculated. E.g. 0: The local environment around the
    # initial position before the HAT reaction takes place. 5: The half-way point of the transition, i.e. exactly
    # at the position between initial and end position, interpolated through a straight line. 10: the final position
    i_position=i_position,
    # Parameters used for the SOAP feature creation
    soap_config=dict(
        r_cut=2.5,
        n_max=12,
        l_max=12,
        sigma=0.3,
        sparse=False,
        species=["H", "C", "O", "N", "S"],
    ),
    # By default three atoms are placed at the initial position, mid-way position and end position of the HAT hydrogen.
    # To distinguish between them and from the rest of the atoms, we give them fictitious elements X, V and Y respectively.
    # SOAP does not care specifically what the elements are only that they are different
    # The elements themselves can be changed, e.g. make all three to hydrogen: H_start="H", H_mid="H", H_end="H".
    # If you do not wish to add the additional mid and end HAT atoms, then use 'H_mid=None, H_end=None'
    fictitious_HAT_species=dict(H_start="X", H_mid="V", H_end="Y"),
    normalise_S=True,  # Whether to normalize the feature vectors
    S_epsilon=1e-8,  # Add small number to each component in case of a null vector
    hashid=hashid,  # Each run has a unique automatically generated hash for later identification
    config_save=str(
        (cwd / "soap" / f"config.txt").resolve()
    ),  # Save the creation config to a file
    SD_save=str(
        (cwd / "soap" / f"SOAP_dist_at_{i_position}.npz").resolve()
    ),  # Where to save the final feature & distances
    # Whether to save the features themselves. For typical covariance functions this is not needed, since they only rely on the distance
    save_soap=False,
    # Whether to calculate the distance matrix between the features
    calc_D=True,
    # Start time for later information
    startdatetime=datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
    # Set to True to use all transitions, else only use a few for testing
    production_run=True,
    df=str((cwd / "atoms.pkl").resolve()),  # were the data is located
    log_level=logging.DEBUG,  # log level
)

log = logging.getLogger(__name__)
logging.basicConfig(level=config["log_level"])

log.info("Running with config:")
log.info(pformat(config))

log.info(f"Started run: {hashid}")

log.info("Load dataset")
df = pd.read_pickle(config["df"])

if not config["production_run"]:
    df = df.iloc[:100]
    print(f"WARNING: this is a TEST RUN with only {df.shape[0]} transitions")

log.debug("Loaded dataset")
n_systems = df.shape[0]

# Creates SOAP constructor
config["soap_config"]["species"] = np.unique(
    config["soap_config"]["species"] + list(config["fictitious_HAT_species"].values())
)
soaper = SOAP(**config["soap_config"])
n_features = soaper.get_number_of_features()  # length of soap vector
log.info(f"SOAP n features: {n_features}")


def get_soap_position(i_position, start, end):
    """3D Position of the HAT transition step, i.e. i_position=0->start, i_position=10->end, i_position=5->middle"""
    return (start + (i_position / 10.0) * (end - start)).copy()


# Loop though each reaction (here one reaction has two directions).
# For each reaction direction add the fictitious HAT hydrogen atoms and add the system to the collector.
# Also add the 3D position of the SOAP center in each case

collector = []  # list which collects the created systems and metadata
for i, (index, row) in enumerate(df.iterrows()):
    transition_hash = row.transition_hash
    if i % 1000 == 0:
        log.debug(f"Transition: {i}")

    r_H = np.array(
        row.r_H
    )  # list of the positions the HAT hydrogen undergoes r_H[0] is start, r_H[-1] (=r_H[10]) is the end
    assert not np.allclose(r_H[0], r_H[-1])  # sanity check that hydrogen atom moves
    r_H_start = r_H[0].copy()
    r_H_end = r_H[-1].copy()
    atoms = row.atoms.copy()

    # If the mid and final HAT atoms are added, then the first atom will be the hydrogen atom at the start,
    # the second last atom will be the HAT atom in the middle and the last atom will be the HAT atom at the final position

    if config["fictitious_HAT_species"]["H_mid"]:
        atoms.append(Atom(config["fictitious_HAT_species"]["H_mid"], r_H[5]))

    if config["fictitious_HAT_species"]["H_end"]:
        atoms.append(Atom(config["fictitious_HAT_species"]["H_end"], r_H[-1]))

    chemical_symbols = atoms.get_chemical_symbols()
    if config["fictitious_HAT_species"]["H_start"]:
        chemical_symbols[0] = config["fictitious_HAT_species"]["H_start"]
        atoms.set_chemical_symbols(chemical_symbols)
    else:
        assert chemical_symbols[0] == "X"

    positions = atoms.get_positions()

    # Forward direction
    positions[0] = r_H[
        0
    ].copy()  # set the first atoms position to the initial HAT position

    atoms.set_positions(positions)
    assert np.allclose(atoms.get_positions()[0], r_H[0])  # sanity check

    # center of local environment
    soap_position = get_soap_position(config["i_position"], start=r_H[0], end=r_H[-1])

    # Add the system and center to the collector
    collector.append([atoms.copy(), [soap_position], transition_hash, "forward"])

    # Backward direction
    positions[0] = r_H[
        -1
    ].copy()  # now the HAT atom starts at the original final position

    if config["fictitious_HAT_species"]["H_end"]:
        positions[-1] = r_H[
            0
        ].copy()  # the HAT atom ends at the original initial position
    atoms.set_positions(positions)
    assert np.allclose(atoms.get_positions()[0], r_H[-1])  # sanity check

    # center of local environment, notice start and end are reversed
    soap_position = get_soap_position(config["i_position"], start=r_H[-1], end=r_H[0])

    # Notice that the mid-transition HAT atom was not changed

    # Add the system and center to the collector
    collector.append([atoms.copy(), [soap_position], transition_hash, "backward"])

    # Sanity checks
    atoms_forward = collector[-2][0].copy()
    atoms_backward = collector[-1][0].copy()
    soap_position_forward = collector[-2][1][0].copy()
    soap_position_backward = collector[-1][1][0].copy()

    if config["fictitious_HAT_species"]["H_start"]:
        assert atoms_forward.get_chemical_symbols()[0] == config[
            "fictitious_HAT_species"
        ]["H_start"] and np.allclose(r_H[0], atoms_forward.get_positions()[0])
        assert atoms_backward.get_chemical_symbols()[0] == config[
            "fictitious_HAT_species"
        ]["H_start"] and np.allclose(r_H[-1], atoms_backward.get_positions()[0])
    else:
        assert atoms_forward.get_chemical_symbols()[0] == "H" and np.allclose(
            r_H[0], atoms_forward.get_positions()[0]
        )
        assert atoms_backward.get_chemical_symbols()[0] == "H" and np.allclose(
            r_H[-1], atoms_backward.get_positions()[0]
        )

    if config["fictitious_HAT_species"]["H_mid"]:
        assert atoms_forward.get_chemical_symbols().copy()[-2] == config[
            "fictitious_HAT_species"
        ]["H_mid"] and np.allclose(r_H[5], atoms_forward.get_positions()[-2])
        assert atoms_backward.get_chemical_symbols().copy()[-2] == config[
            "fictitious_HAT_species"
        ]["H_mid"] and np.allclose(r_H[5], atoms_backward.get_positions()[-2])

    if config["fictitious_HAT_species"]["H_end"]:
        assert atoms_forward.get_chemical_symbols().copy()[-1] == config[
            "fictitious_HAT_species"
        ]["H_end"] and np.allclose(r_H[-1], atoms_forward.get_positions()[-1])
        assert atoms_backward.get_chemical_symbols().copy()[-1] == config[
            "fictitious_HAT_species"
        ]["H_end"] and np.allclose(r_H[0], atoms_backward.get_positions()[-1])

    assert np.allclose(soap_position_forward + soap_position_backward - r_H[0], r_H[-1])
    assert np.allclose(r_H_start, r_H[0]) and np.allclose(r_H_end, r_H[-1])

atoms_collect = [a[0] for a in collector]
positions = [a[1] for a in collector]
transition_hashes = np.array([a[2] for a in collector])
directions = np.array([a[3] for a in collector])

n_systems = len(atoms_collect)
log.debug(f"N systems: {n_systems}")

# Calculate the feature vectors for each collected system
log.info("Calculate Features (usually takes <1 min)")
tic_S = time.time()
S = soaper.create(atoms_collect, positions, n_jobs=1, verbose=False)
S = np.squeeze(
    S.copy(), axis=1
)  # Remove redundant dimension, since only one SOAP vector per system
t_S = time.time() - tic_S

log.debug(S)
log.debug(f"S: {S.nbytes * 1e-6:.3f} MB in {t_S:.3f}s, shape: {S.shape}")

# Perform normalization if requested
if config["normalise_S"]:
    log.info("Normalise S (~ couple of seconds)")
    tic_S_norm = time.time()
    S += config[
        "S_epsilon"
    ]  # add small epsilon to each component in case of a null vector
    norm_S = np.linalg.norm(S, axis=1)[..., None]
    S = S / norm_S
    t_S_norm = time.time() - tic_S_norm
    log.debug(f"Normalised in {t_S_norm:.3f}s")
    # Sanity check
    assert (
        np.abs(np.linalg.norm(S[0, :]) - 1.0) < 1e-4
    ), f"Not normalised: {np.linalg.norm(S[0])}"
else:
    log.warning("Not normalising features as requested")

# Compute the distance matrix of each feature vector with each other feature vector
log.info("Calc D")
if config["calc_D"]:
    log.info("Start D calc (grab a coffee, this may take a couple of minutes)")
    tic_D = time.time()
    D = pairwise_distances(S, S, n_jobs=-1)
    t_D = time.time() - tic_D
    log.debug(D)
    log.debug(f"D: {D.nbytes * 1e-6:.3f} MB in {t_D:.3f}s, shape: {D.shape}")
else:
    log.warning("Calculation of distances skipped per request")
    D = None

log.debug(
    f"Save S, D and config at {config['SD_save']} for position: {config['i_position']}"
)
if not config["save_soap"]:
    log.info("SOAP features not saved as requested")
    S = None  # drop the calculated features

np.savez(
    config["SD_save"],
    HAT_step_position=i_position,
    S=S,
    D=D,
    transition_hashes=np.array(transition_hashes),
    direction=directions,
    config=config,
    hashid=hashid,
)

with open(config["config_save"], "a+") as f:
    f.write(
        f"Config: {pformat(config, indent=4)}\n\n"
        f"------------------------------------------------------------------------------------------------\n\n"
    )
