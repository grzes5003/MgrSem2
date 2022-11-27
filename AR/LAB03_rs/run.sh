#!/bin/bash -l
#SBATCH -J MPI_AR_LAB04_GK
#SBATCH -N 1
#SBATCH --ntasks-per-node=12
#SBATCH --time=00:25:00
#SBATCH -A plgar2022-cpu
#SBATCH -p plgrid
#SBATCH --output="output.out"
#SBATCH --error="error.err"

if [ -z "$SCRIPT" ]; then
  TODAY=$(date +"%d_%H_%M")
  exec 3>&1 4>&2
  trap 'exec 2>&4 1>&3' 0 1 2 3
  exec 1>log_"$TODAY".log 2>&1
fi

module load rust/1.63.0-gcccore-10.3.0
module load openmpi/4.1.2-intel-compilers-2021.4.0
module load clang

echo "Compiling LAB03_rs"

cargo update
cargo build --release

echo "Starting LAB03_rs"

prog=./target/release/LAB03_rs

ITER=1000
N=10000

for ((iter = 3; iter > 0; iter--)); do
  for ((n_size = 100; n_size <= N; n_size *= 10)); do
    mpiexec -np 1 "$prog" --it "$ITER" -n "$n_size"
    for ((threads = 2; threads <= 12; threads += 2)); do
      mpiexec -np "$threads" "$prog" --it "$ITER" -n "$n_size"
    done
  done
done

echo $?
