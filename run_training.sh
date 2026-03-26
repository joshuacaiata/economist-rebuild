#!/bin/bash
#SBATCH --job-name=mobile_agents
#SBATCH --account=def-klarson
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=100G
#SBATCH --mail-user=jcaiata.slurm@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=slurm-%j.out

module purge
module load python/3.11

# Create venv if it doesn't exist
if [ ! -d "$SLURM_TMPDIR/env" ]; then
    python -m venv "$SLURM_TMPDIR/env"
    source "$SLURM_TMPDIR/env/bin/activate"
    pip install --no-index -r requirements.txt
else
    source "$SLURM_TMPDIR/env/bin/activate"
fi

python main.py --config config_mobile_only.yaml --training-type two_phase --phase 1 --use-gpu
