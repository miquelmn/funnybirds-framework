#!/bin/bash
#SBATCH --job-name=funnyB_all        # Job name
#SBATCH --nodes=1                    # Number of nodes
#SBATCH --ntasks=1                   # Number of tasks
#SBATCH --cpus-per-task=12           # Number of CPU cores per task
#SBATCH --gres=gpu:1                 # Request 1 GPU
#SBATCH --mem=16G                    # Requested RAM memory
#SBATCH --output=logs/%x_%j.out      # Standard output file in the logs subfolder
#SBATCH --error=logs/%x_%j.err       # Standard error file in the logs subfolder

uv sync
source .venv/bin/activate

# Runs every explainer in the EXPLAINERS dictionary of evaluate_all_explainers.py and every
# protocol. Rows are appended to the TSV; --skip_done skips explainers already completed.
python evaluate_all_explainers.py --data ./data/FunnyBirds --model vgg16 --checkpoint_name ./checkpoints/vgg16_final_1_checkpoint_best.pth.tar --output results/vgg16.tsv --gpu 0 --skip_done

echo "All tasks completed successfully."
