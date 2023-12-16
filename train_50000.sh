#!/bin/sh

#SBATCH --job-name=train_model2
#SBATCH --account=minjilab0
#SBATCH --partition=gpu,spgpu
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=20GB
#SBATCH --time=70:00:00
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=mdolo@umich.edu

echo "training model with 50,000 dataset"

time {
	python autoencoder.py ML_molecular_data_50000.tsv train_50000.out
}
