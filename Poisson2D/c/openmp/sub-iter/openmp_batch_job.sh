#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=cpu
#SBATCH --account=p200117
#SBATCH --qos=default
#SBATCH --time=00:05:00

#Load GCC module
module load GCC

echo "===Compile Phase==="
#Clean before compile
make clean

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

#Compile C program with gcc
make

echo "===Execution Phase==="
#Execute the program
./main.out

echo "===END==="
