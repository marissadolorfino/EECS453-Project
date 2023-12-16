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
#SBATCH -o full_original_model.out

echo "training on dataset of 200,000, trying to print out outputs"

time {
	python sigmoid_autoencoder.py ML_molecular_data.tsv outputs_fulldata_orig_sigmoid.csv
}
