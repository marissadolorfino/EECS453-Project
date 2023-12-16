#!/bin/sh

#SBATCH --job-name=train_model
#SBATCH --account=minjilab0
#SBATCH --partition=spgpu,gpu
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=20GB
#SBATCH --time=70:00:00
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=mdolo@umich.edu

echo "training on dataset of 100, trying to print out outputs"

time {
	python autoencoder.py ML_molecular_data_truncated.tsv training_outputs100.tsv
}
