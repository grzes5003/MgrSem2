#!/bin/bash -l


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



echo $?