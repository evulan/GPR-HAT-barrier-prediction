"""Helper functions especially to create unique transition hashes for comparison"""

import numpy as np
import hashlib
import logging


def init_logger(level):
    """Create a colored logger for different log levels"""

    class CustomFormatter(logging.Formatter):

        color_debug = "\033[90m"
        color_info = "\033[38;5;246m"
        color_warning = "\033[93m"
        color_error = "\033[31m"
        color_critical = "\033[91m"

        format = "%(levelname)8s %(asctime)s: %(message)s"

        logger_formats = {
            logging.DEBUG: color_debug + format + "\x1b[0m",
            logging.INFO: color_info + format + "\x1b[0m",
            logging.WARNING: color_warning + format + "\x1b[0m",
            logging.ERROR: color_error + format + "\x1b[0m",
            logging.CRITICAL: color_critical + format + "\x1b[0m",
        }

        def format(self, record):
            formatter = logging.Formatter(
                self.logger_formats.get(record.levelno), datefmt="%H:%M:%S"
            )
            return formatter.format(record)

    if level == "debug":
        level_logging = logging.DEBUG
    elif level == "info":
        level_logging = logging.INFO
    elif level == "warning":
        level_logging = logging.WARNING
    elif level == "error":
        level_logging = logging.ERROR
    else:
        raise ValueError("Invalid log level")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(CustomFormatter())
    logger = logging.getLogger(__name__)
    logger.setLevel(level_logging)
    logger.addHandler(stream_handler)
    return logger


def strings_to_hash(strings):
    """Merge multiple strings into single string and create a single hash"""
    super_string = ":::".join(strings)
    return hashlib.md5(super_string.encode()).hexdigest()


def nparray_to_string(arr):
    """Create a string from a numpy array in standardized form"""
    return np.array2string(
        arr,
        max_line_width=10000,
        precision=3,
        suppress_small=False,
        separator=",",
        threshold=10000,
        floatmode="fixed",
        sign="+",
    )


def transition_to_hash(start_atoms, end_atoms, energies):
    """Create a unique hash of a reaction from positions, energies etc"""
    pos_start_str = nparray_to_string(start_atoms.get_positions())
    element_start_str = str(start_atoms.symbols)
    pbc_start_str = nparray_to_string(start_atoms.get_pbc())

    pos_end_str = nparray_to_string(end_atoms.get_positions())
    element_end_str = str(end_atoms.symbols)
    pbc_end_str = nparray_to_string(end_atoms.get_pbc())

    energies_str = nparray_to_string(energies)

    return strings_to_hash(
        [
            pos_start_str,
            element_start_str,
            pbc_start_str,
            pos_end_str,
            element_end_str,
            pbc_end_str,
            energies_str,
        ]
    )


def clean_atoms_object(atoms):
    """Remove unnecessary configs from atoms object"""
    atoms.set_pbc(False)
    atoms.set_cell([0, 0, 0])
    atoms.set_array("bfactor", None)
    atoms.set_array("atomtypes", None)
    atoms.set_array("occupancy", None)
    atoms.set_array("residuenames", None)
    atoms.set_array("residuenumbers", None)


def same_arr(x, y, eps=1e-7):
    """Check if arrays are almost equal"""
    return np.linalg.norm(x - y) < eps
