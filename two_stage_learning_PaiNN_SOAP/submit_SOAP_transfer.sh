#!/bin/bash
#SBATCH --job-name=SOAP_two_stage
#SBATCH --time=02:00:00
#SBATCH --output=./results/slurm_two_stage_gpr_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --array=0-7

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
cd two_stage_learning_PaiNN_SOAP || exit
pwd
echo "------------------------ Start ------------------------"
python SOAP_second_stage.py "$SLURM_ARRAY_TASK_ID" || exit
end=$(date +"%T")
echo "------------------------ End time: $end ------------------------"
