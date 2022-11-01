#!/bin/bash -l
#SBATCH -J MPISieve
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --time=00:05:00
#SBATCH -A plgar2022-cpu
#SBATCH -p plgrid
#SBATCH --output="output.out"
#SBATCH --error="error.err"

srun /bin/hostname

if [ -z "$SCRIPT" ]; then
  TODAY=$(date +"%d_%H_%M")
  exec 3>&1 4>&2
  trap 'exec 2>&4 1>&3' 0 1 2 3
  exec 1>log_"$TODAY".log 2>&1
fi

module load scipy-bundle/2021.10-intel-2021b
module load openmpi/4.1.4-gcc-11.3.0
module load cmake/3.23.1-gcccore-11.3.0

echo "Compiling LAB02"

cmake .
make LAB02

echo "Starting LAB02"

mpiexec -np 4 ./LAB02 10000