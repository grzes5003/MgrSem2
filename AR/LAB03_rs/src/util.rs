use std::io::BufRead;
use mpi::Rank;
use mpi::topology::SystemCommunicator;
use mpi::traits::{Communicator, Destination, Source};
use mpi::traits::Equivalence;
use clap::Parser;
use rand::Rng;

use crate::{Matrix, Procs};

type Float = f32;

#[derive(Equivalence)]
pub struct Pair(pub Procs, pub Procs);

pub struct CommConfig<'a> {
    comm: &'a SystemCommunicator,
    rank: Procs,
    p: Procs,
    grid_len: Procs,
}

impl<'a> CommConfig<'a> {
    pub fn unpack(&self) -> (
        &'a SystemCommunicator,
        Procs, Procs, Procs
    ) {
        (self.comm, self.rank, self.p, self.grid_len)
    }

    pub fn new(comm: &'a SystemCommunicator,
               rank: Procs, p: Procs, grid_len: Procs) -> Self {
        CommConfig {
            comm,
            rank,
            p,
            grid_len
        }
    }
}

pub fn get_slices(comm_config: &CommConfig) -> Pair {
    let (comm, rank, p, grid_len) = comm_config.unpack();
    match rank {
        0 => {
            let mut length_per_core = grid_len / p;
            let mut begin = 1;
            let end = length_per_core as Procs;
            let mut current = 1 + length_per_core;

            for core in 1..p {
                let mut indexes: Pair = Pair(current as Procs, (current + length_per_core - 1) as Procs);
                comm.process_at_rank(core as Rank)
                    .send(&indexes);
                current += length_per_core;
            }
            Pair(begin, end)
        }
        _ => {
            let (indexes, _) = comm.process_at_rank(0)
                .receive::<Pair>();
            indexes
        }
    }
}

pub fn gen_matrix<F>(x: usize, y: usize, generator: F) -> Vec<Vec<Float>>
    where F: FnOnce() -> Float + Copy {
    (0..x).into_iter()
        .map(|_| (0..y).map(|_| generator()).collect())
        .collect()
}

pub fn update_matrix(comm_config: &CommConfig,
                     matrix: Matrix, matrix_tmp: &mut Matrix,
                     mut up_slice: Vec<Float>, mut down_slice: Vec<Float>, pair: Pair
) -> Matrix {
    let (comm, rank, p, grid_len) = comm_config.unpack();
    let (begin, end) = (pair.0, pair.1);

    if rank % 2 == 0 {
        if rank > 0 {
            comm.process_at_rank((rank - 1) as Rank)
                .send(&matrix[0][..]);
        }
        if rank < p - 1 {
            let res = comm.process_at_rank((rank + 1) as Rank)
                .receive_vec();
            up_slice = res.0;
            comm.process_at_rank((rank + 1) as Rank)
                .send(&matrix[end - begin][..]);
        }
        if rank > 0 {
            let res = comm.process_at_rank((rank - 1) as Rank)
                .receive_vec();
            down_slice = res.0;
        }
    } else {
        if rank < p - 1 {
            let res = comm.process_at_rank((rank + 1) as Rank)
                .receive_vec();
            up_slice = res.0;
        }
        if rank > 0 {
            comm.process_at_rank((rank - 1) as Rank)
                .send(&matrix[0][..]);
            let res = comm.process_at_rank((rank - 1) as Rank)
                .receive_vec();
            down_slice = res.0;
        }
        if rank < p - 1 {
            comm.process_at_rank((rank + 1) as Rank)
                .send(&matrix[end - begin][..]);
        }
    }


    let (mut new_down, mut new_up) = (0f32, 0f32);
    for y in begin..(end + 1) {
        for x in 1..(grid_len + 1) {
            if y - 1 == 0 {
                new_down = 0f32;
            } else if y - 1 >= begin {
                new_down = matrix[y - 1 - begin][x];
            } else {
                new_down = down_slice[x];
            }

            if y + 1 == grid_len + 1 {
                new_up = 0f32;
            } else if y + 1 <= end {
                new_up = matrix[y + 1 - begin][x]
            } else {
                new_up = up_slice[x]
            }
            matrix_tmp[y - begin][x] = (matrix[y - begin][x - 1] + matrix[y - begin][x + 1]
                + new_up + new_down) / 4f32;
        }
    }
    matrix_tmp.to_owned()
}

pub fn join(comm_config: &CommConfig, matrix: Matrix, result: &mut Matrix) {
    let (comm, rank, p, grid_len) = comm_config.unpack();

    if rank == 0 {
        let mut idx = 0;
        for row in matrix {
            if row[1] == 0f32 {
                break;
            }
            result[idx] = row;
            idx += 1;
        }

        for i in 0..(p - 1) {
            let res = comm.process_at_rank((i + 1) as Rank)
                .receive_vec::<Float>();
            let tmp: Matrix = res.0.chunks(grid_len + 2)
                .map(|v| v.to_vec())
                .collect();

            for x in tmp {
                if x[1] == 0f32 {
                    break;
                }
                result[idx] = x;
                idx += 1;
            }

        }
    } else {
        let flatten_matrix: Vec<Float> = matrix.into_iter().flatten().collect();
        comm.process_at_rank(0)
            .send(&flatten_matrix[..])
    }
}