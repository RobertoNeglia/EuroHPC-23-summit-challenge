#!/bin/bash -l
#SBATCH -N 1
#SBATCH --ntasks=5
#SBATCH --cpus-per-task=1
#SBATCH -p cpu
#SBATCH -q test
#SBATCH --time 00:05:00

#Load OpenMPI module
module load OpenMPI

echo "===Compile Phase==="
#Clean before compile
make clean

#Compile C program with mpicc
make

echo "===Execution Phase==="
#Execute the program
srun ./main

echo "===END==="
