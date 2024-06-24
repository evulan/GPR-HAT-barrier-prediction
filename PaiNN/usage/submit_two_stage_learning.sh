#!/bin/bash
#SBATCH --job-name=PaiNN_two_stage
#SBATCH --time=06:00:00
#SBATCH --array=0-9
#SBATCH --output=./results/two_stage_learning/slurm_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=30G
#SBATCH --gres=gpu:1

start=$(date +"%T")
echo "------------------------ Start Rank $SLURM_ARRAY_TASK_ID at: $start ------------------------"
echo "------------------------ Load environment variables ------------------------"
ENVFILE="../../local.env"
source $ENVFILE || exit
echo "------------------------ Go to Root ------------------------"
cd "$PROJECTROOT" || exit
pwd
echo "------------------------ Load Conda ------------------------"
eval "$($CONDABIN shell.bash hook)" || exit
echo "------------------------ Activate Env ------------------------"
conda activate painn_env || exit
echo "------------------------ Go to folder ------------------------"
cd PaiNN/usage || exit
pwd
echo "------------------------ Start ------------------------"
python two_stage_learning.py "$SLURM_ARRAY_TASK_ID" || exit
end=$(date +"%T")
echo "------------------------ End time: $end ------------------------"
