#!/usr/bin/env bash

echo "This will install 3 environments needed for the SOAP GPR, PaiNN models and MGK GPR respectively"
echo "This can take a while (>30 min)."
echo "Make sure that ../local.env has CONDABIN & OPENMPIMODULE set."

read -p "Confirm? (y/n) " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then

    echo "Load custom variables"

    # Make sure CONDABIN is defined in ENVFILE
    ENVFILE="../local.env"
    source $ENVFILE || exit
    echo "Conda Bin: $CONDABIN"
    echo "OpenMPI Module: $OPENMPIMODULE"

    # Initialize conda
    echo "Initialize Conda"
    eval "$($CONDABIN shell.bash hook)" || exit

    ##### Main & SOAP GPR - Start #####
    echo 'Installing main environment: main_gpr_env'
    conda env create -f main_gpr_env/environment.yaml || exit
    conda activate main_gpr_env || exit
    pip install -r main_gpr_env/requirements_jax.txt || exit
    pip install --force-reinstall -r main_gpr_env/requirements.txt || exit
    conda deactivate || exit
    echo 'Installed main environment: main_gpr_env'
    ##### Main & SOAP GPR - End #####

    ##### PaiNN - Start #####
    echo 'Installing PaiNN environment: painn_env'
    # Install conda environment for PaiNN model
    conda env create -f painn_env/environment.yaml || exit
    conda activate painn_env || exit

    # Add conda path automatically to path on every activation, in order to use cuda
    mkdir -p "$CONDA_PREFIX"/etc/conda && cd "$_" || exit
    mkdir -p activate.d deactivate.d || exit
    echo '#!/bin/sh' >activate.d/env_vars.sh || exit
    echo "export OLD_LD_LIBRARY_PATH=${LD_LIBRARY_PATH}" >>activate.d/env_vars.sh || exit
    echo "export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}" >>activate.d/env_vars.sh || exit
    source activate.d/env_vars.sh || exit
    echo '#!/bin/sh' >deactivate.d/env_vars.sh || exit
    echo "LD_LIBRARY_PATH=${OLD_LD_LIBRARY_PATH} unset OLD_LD_LIBRARY_PATH" >>deactivate.d/env_vars.sh || exit
    cd - || exit

    # Now install pip packages
    pip install -r painn_env/requirements.txt || exit

    # Install custom PaiNN package developed by Kai Riedmiller
    cd ../PaiNN/barrier_gnn || exit
    pip install -e ./ || exit
    cd - || exit
    conda deactivate || exit
    echo 'Installed PaiNN environment: painn_env'
    ##### PaiNN - End #####

    #### MGK GPR - Start #####
    echo 'Installing MGK environment: mgk_gpr_env'
    conda env create -f mgk_gpr_env/environment.yaml || exit
    conda activate mgk_gpr_env || exit
    module load "$OPENMPIMODULE" || exit
    pip install mpi4py==3.1.4 || exit

    # Add conda path automatically to path on every activation, in order to use cuda
    mkdir -p "$CONDA_PREFIX"/etc/conda && cd "$_" || exit
    mkdir -p activate.d deactivate.d || exit
    echo '#!/bin/sh' >activate.d/env_vars.sh || exit
    echo "export OLD_LD_LIBRARY_PATH=${LD_LIBRARY_PATH}" >>activate.d/env_vars.sh || exit
    echo "export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}" >>activate.d/env_vars.sh || exit
    source activate.d/env_vars.sh || exit
    echo '#!/bin/sh' >deactivate.d/env_vars.sh || exit
    echo "LD_LIBRARY_PATH=${OLD_LD_LIBRARY_PATH} unset OLD_LD_LIBRARY_PATH" >>deactivate.d/env_vars.sh || exit
    cd - || exit

    # Now install jax
    pip install -r mgk_gpr_env/requirements_jax.txt || exit
    conda deactivate || exit
    echo 'Installed MGK environment: mgk_gpr_env'
#### MGK GPR - End #####

fi
