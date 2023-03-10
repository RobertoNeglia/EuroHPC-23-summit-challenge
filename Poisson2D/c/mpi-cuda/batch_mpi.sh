#!/bin/bash -l
#SBATCH --nodes=4
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --qos=default
#SBATCH --time 00:05:00
#SBATCH --account=p200117

#Load OpenMPI module
module load OpenMPI
module load NVHPC
mpicc --version
nvcc --version

echo "===Clean Phase==="
#Clean before compile
rm *.out *.o

echo "===Compile Phase==="
#Compile C program with mpicc
mpicc -g -Wall -O3 -c main.c solver.c
#Compile CUDA program with nvcc
nvcc -c solve.cu
echo "===Link Phase==="
#Link the two together
mpicc main.o solver.o solve.o -lcudart -lm -o main.out

echo "===Execution Phase==="
#Execute the program
srun ./main.out

echo "===END==="
