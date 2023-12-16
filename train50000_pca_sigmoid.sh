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
#SBATCH -o train50000_pca_sigmoid.out

echo "training model with 50,000 dataset and pca reduced inputs and sigmoid"

time {
	python pca_then_train_autoencoder_sigmoid.py ML_molecular_data_50000.tsv outputs_train50000_pca_sigmoid.csv
}

