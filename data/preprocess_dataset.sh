#!/usr/bin/env bash

echo "Make sure that you"
echo "* Installed the conda environments with ../install/install.sh"
echo "* Downloaded the trajectory dataset from https://heidata.uni-heidelberg.de/file.xhtml?persistentId=doi:10.11588/data/TGDD4Y/RXRYK8&version=2.0 and placed the dataset_traj.zip into data/pdb"
echo "* Downloaded the synthetic dataset from https://heidata.uni-heidelberg.de/file.xhtml?persistentId=doi:10.11588/data/TGDD4Y/LIOOQN&version=2.0 and placed the dataset_synth.zip into data/pdb"
echo "* Downloaded the metadata.csv file from https://heidata.uni-heidelberg.de/file.xhtml?persistentId=doi:10.11588/data/TGDD4Y/TPD3XK&version=2.0 and placed it into data/pdb"
read -p "Confirm? (y/n) " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then

    cd pdb
    ENVFILE="../../local.env"
    source $ENVFILE || exit
    echo "Initialize Conda: $CONDABIN"
    eval "$($CONDABIN shell.bash hook)" || exit

    echo "Unzip and rename"
    unzip dataset_synth.zip && mv dataset_2208_synth synth
    unzip dataset_traj.zip && mv dataset_2208_traj traj

    echo "Create a pandas dataframe from the pdb files"
    cd ..
    conda activate main_gpr_env || exit
    python pdb_to_atoms.py || exit

    echo "Done"

fi
