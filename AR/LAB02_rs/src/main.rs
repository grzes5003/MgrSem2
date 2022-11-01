#![allow(warnings)]

use std::env;
use std::ops::Not;
use mpi::collective::SystemOperation;
use mpi::traits::*;
use crate::sieve::{BLOCK_STEP, BLOCK_FIRST,
                   Procs, block_low, block_high,
                   block_size, setup_primes_arr};

mod sieve;

fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let root_rank = 0;

    let t_start = mpi::time();
    world.barrier();

    let n = env::args().nth(1)
        .unwrap_or("10000".to_string())
        .parse::<Procs>()
        .unwrap();
    let p = world.size() as Procs;
    let id = world.rank() as Procs;

    let low_value = BLOCK_FIRST + block_low(id, p, n - 1) * BLOCK_STEP;
    let high_value = BLOCK_FIRST + block_high(id, p, n - 1) * BLOCK_STEP;
    let size = block_size(id, p, n - 1);

    let sqrt_n = f32::sqrt(n as f32) as usize;
    let mut primes = setup_primes_arr(sqrt_n);
    let mut marked = vec![0 as u8; size];

    let num_per_block = 1024 * 1024;
    let mut block_low_value = low_value;
    let mut block_high_value = Procs::min(high_value,
                                          low_value + num_per_block * BLOCK_STEP);

    let mut first;
    let mut first_value_index;

    for _ in (0..size)
        .step_by(num_per_block as usize) {
        for prime in 3..sqrt_n {
            if primes[prime] == 1 {
                continue;
            }
            if prime * prime > block_low_value as usize {
                first = prime * prime;
            } else {
                if block_low_value % prime as Procs == 0 {
                    first = block_low_value as usize;
                } else {
                    first = prime - (block_low_value % prime as Procs) as usize +
                        block_low_value as usize;
                }
            }

            if ((first + prime) & 1) != 0 {
                first += prime;
            }

            first_value_index = (first - BLOCK_FIRST as usize) / BLOCK_STEP as usize -
                block_low(id, p, n - 1);
            let prime_doubled = prime << 1;
            let prime_step = prime_doubled / BLOCK_STEP;
            for i in (first..=high_value).step_by(prime_doubled) {
                marked[first_value_index] = 1;
                first_value_index += prime_step;
            }
        }

        block_low_value += num_per_block * BLOCK_STEP;
        block_high_value = Procs::min(high_value,
                                      block_high_value + num_per_block * BLOCK_STEP);
    }

    let count = marked.into_iter()
        .filter(|val| *val == 0)
        .count();

    let rank = world.rank();
    if rank == root_rank {
        let mut sum: Procs = 0;
        world
            .process_at_rank(root_rank)
            .reduce_into_root(&count, &mut sum, SystemOperation::sum());
        world.barrier();
        let t_end = mpi::time();
        println!("t={};{};n={}", t_end - t_start, sum, n);
    } else {
        world
            .process_at_rank(root_rank)
            .reduce_into(&count, SystemOperation::sum());
        world.barrier();
    }
}