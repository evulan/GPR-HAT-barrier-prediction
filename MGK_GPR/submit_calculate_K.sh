#!/bin/bash
#SBATCH --job-name=GD_K
#SBATCH --time=00:45:00
#SBATCH --output=./results/slurm_calc_K.out
#SBATCH --nodes=10
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

start=$(date +"%T")
echo "------------------------ Start time: $start ------------------------"
echo "------------------------ Load environment variables ------------------------"
ENVFILE="../local.env"
source $ENVFILE || exit
echo "------------------------ Go to Root ------------------------"
cd "$PROJECTROOT" || exit
pwd
echo "------------------------ Load MPI ------------------------"
module load OpenMPI/3.1.4-GCC-8.3.0 || exit
echo "------------------------ Export MPI ------------------------"
MPICC="$(which mpicc)" || exit
export MPICC || exit
echo "------------------------ Load Conda ------------------------"
eval "$($CONDABIN shell.bash hook)" || exit
echo "------------------------ Activate Env ------------------------"
conda activate mgk_gpr_env || exit
echo "------------------------ Go to folder ------------------------"
cd MGK_GPR || exit
echo "------------------------ Start ------------------------"
mpiexec python create_K.py || exit
end=$(date +"%T")
echo "------------------------ End time: $end ------------------------"
