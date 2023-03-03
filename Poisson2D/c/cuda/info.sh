#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=2
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --partition=gpu
#SBATCH --account=p200117
#SBATCH --qos=default


#Load NVHPC module
module load NVHPC
#module load OpenMPI 
#test purpose openmpi load 

echo "===Compile Phase==="
#Clean before compile
make clean

#Compile C program with nvcc (in parallel)
make

echo "===Execution Phase==="
#Execute the program
./gpu_info.out

echo "===END==="
