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
#SBATCH -o pca.out

echo "pca on test set of 100 data examples"

time {
	python pca_on_inputs.py ML_molecular_data_500.tsv reduced_dataset_test.csv
}
