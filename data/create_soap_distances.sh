#!/usr/bin/env bash

echo "Make sure that you ran preprocess_dataset.sh previously"
read -p "Confirm? (y/n) " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then

    ENVFILE="../local.env"
    source $ENVFILE || exit
    echo "Initialize Conda: $CONDABIN"
    eval "$($CONDABIN shell.bash hook)" || exit
    conda activate main_gpr_env || exit

    echo "Create SOAP vectors and their distances at the start positions S (step: 0)"
    python atoms_to_soap.py 0 || exit

    echo "Create SOAP vectors and their distances at the start positions M (step: 5)"
    python atoms_to_soap.py 5 || exit

    echo "Create SOAP vectors and their distances at the start positions E (step: 10)"
    python atoms_to_soap.py 10 || exit

    echo "Done"

fi
