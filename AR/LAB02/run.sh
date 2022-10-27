#!/bin/bash -l
#SBATCH -J MPISieve
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --mem-per-cpu=128MB
#SBATCH --time=00:05:00
#SBATCH -A plgar2022-cpu
#SBATCH -p plgrid
#SBATCH --output="output.out"
#SBATCH --error="error.err"

srun /bin/hostname
cd "$SLURM_SUBMIT_DIR" || exit

if [ -z "$SCRIPT" ]; then
  TODAY=$(date +"%d_%H_%M")
  exec 3>&1 4>&2
  trap 'exec 2>&4 1>&3' 0 1 2 3
  exec 1>log_"$TODAY".log 2>&1
fi

## Zaladowanie modulu IntelMPI
module add plgrid/tools/impi

echo "Compiling " "$1"

module add plgrid/tools/openmpi
module add plgrid/tools/cmake

cmake .
make "$1"

echo "Starting " "$1"

mpiexec ./lab02 10000