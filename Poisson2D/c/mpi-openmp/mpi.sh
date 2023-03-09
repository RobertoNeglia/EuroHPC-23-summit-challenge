#!/bin/bash -l
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=cpu
#SBATCH --qos=default
#SBATCH --time 00:05:00
#SBATCH --account=p200117

#Load OpenMPI module
module load OpenMPI
mpicc --version

echo "===Compile Phase==="
#Clean before compile
make clean

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

#Compile C program with mpicc
make

echo "===Execution Phase==="
#Execute the program
srun ./main.out

echo "===END==="
