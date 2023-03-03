#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --partition=gpu
#SBATCH --account=p200117
#SBATCH --qos=default


#Load AOCC module
module load NVHPC
nvcc --version

#test purpose openmpi load 
#module load OpenMPI 

echo "===Compile Phase==="
#Clean before compile
make clean

#Compile C program with AOCC (in parallel)
make

echo "===Execution Phase==="
#Execute the program
./main.out

echo "===END==="
