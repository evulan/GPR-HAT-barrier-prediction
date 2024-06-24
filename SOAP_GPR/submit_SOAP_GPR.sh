#!/bin/bash
#SBATCH --job-name=SOAP_fractional
#SBATCH --time=24:00:00
#SBATCH --output=./results/slurm_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=60G
#SBATCH --gres=gpu:1
#SBATCH --array=0,1

start=$(date +"%T")
echo "------------------------ Start time: $start ------------------------"
echo "------------------------ Load environment variables ------------------------"
ENVFILE="../local.env"
source $ENVFILE || exit
echo "------------------------ Go to Root ------------------------"
cd "$PROJECTROOT" || exit
pwd
echo "------------------------ Load Conda ------------------------"
eval "$($CONDABIN shell.bash hook)" || exit
echo "------------------------ Activate Env ------------------------"
conda activate main_gpr_env || exit
echo "------------------------ Go to folder ------------------------"
cd SOAP_GPR || exit
pwd
echo "------------------------ Start ------------------------"
python MAIN_RUN.py "$SLURM_ARRAY_TASK_ID" || exit
end=$(date +"%T")
echo "------------------------ End time: $end ------------------------"
