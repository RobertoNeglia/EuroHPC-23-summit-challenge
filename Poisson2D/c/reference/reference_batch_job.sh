#!/bin/bash -l
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH -p cpu
#SBATCH -q test
#SBATCH --time 00:05:00

#Load AOCC module
module load AOCC

echo "===Compile Phase==="
#Clean before compile
make clean

#Compile C program with clang
make

echo "===Execution Phase==="
#Execute the program
./main

echo "===END==="
