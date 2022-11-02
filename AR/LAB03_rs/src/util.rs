use mpi::Rank;
use mpi::topology::SystemCommunicator;
use mpi::traits::{Communicator, Destination, Source};
use mpi::traits::Equivalence;
use crate::Procs;
use rand::Rng;

type Float = f32;

#[derive(Equivalence)]
pub struct Pair(pub Procs, pub Procs);

pub fn get_slices(comm: &SystemCommunicator, rank: Procs, p: Procs, grid_len: Procs) -> Pair {
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

pub fn gen_matrix(x: usize, y: usize) -> Vec<Vec<Float>> {
    let mut rng = rand::thread_rng();

    (0..x).into_iter()
        .map(|_| vec![rng.gen::<Float>(); y])
        .collect()
}

pub fn update_matrix(comm: &SystemCommunicator, rank: Procs, p: Procs, grid_len: Procs) {

}