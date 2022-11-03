use std::env;
use clap::Parser;
use mpi::collective::CommunicatorCollectives;
use mpi::topology::Communicator;
use ndarray::{Array, array, Array2, Axis};
use ndarray_npy::write_npy;
use crate::util::{gen_matrix, get_slices, join_calc, Pair, update_matrix};
use rand::Rng;
use rand::rngs::ThreadRng;

mod util;

type Procs = usize;
type Matrix = Vec<Vec<f32>>;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    #[arg(long = "it", default_value_t=100)]
    pub iterations: usize,

    #[arg(short = 'n', default_value_t=100)]
    pub grid_len: usize,

    #[arg(long = "save", short='s', default_value_t=false)]
    pub save_results: bool
}

fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let root_rank = 0;

    let t_start = mpi::time();
    world.barrier();

    let p = world.size() as Procs;
    let rank = world.rank() as Procs;

    let args: Args = Args::parse();
    let (grid_len, iterations, save_result) = (
        args.grid_len,
        args.iterations,
        args.save_results);
    let slice_length = (grid_len as f32 / p as f32 + 0.5f32) as Procs + 1;

    let pair: Pair = get_slices(&world, rank, p, grid_len);
    let (begin, end) = (pair.0, pair.1);

    let mut matrix =
        gen_matrix(slice_length, grid_len + 2, || rand::thread_rng().gen::<f32>());

    for y in 0..(end - begin + 1) {
        matrix[y][0] = 0f32;
        matrix[y][grid_len + 1] = 0f32;
    }

    let mut matrix_tmp = gen_matrix(slice_length, grid_len + 2, || 0f32);
    let up_slice = vec![0f32; grid_len as usize + 2];
    let down_slice = vec![0f32; grid_len as usize + 2];
    let mut result = gen_matrix(slice_length, grid_len + 2, || 0f32);


    for _ in 0..iterations {
        matrix = update_matrix(&world, rank, p, grid_len,
                               matrix, &mut matrix_tmp,
                               up_slice.clone(), down_slice.clone(),
                               Pair(begin, end));
    }

    join_calc(&world, rank, p, grid_len, matrix, slice_length, &mut result);
    world.barrier();

    if rank == root_rank {
        // println!("{:?}", result[0]);
        let t_end = mpi::time();
        println!("t={};it={};n={};p={}", t_end - t_start, iterations, grid_len, p);
        if save_result {
            let mut arr = Array2::from_shape_vec(
                (result.len(), result[0].len()),
                result.into_iter().flatten().collect::<Vec<f32>>()).unwrap().to_owned();
            write_npy("results/result.npy", &arr);
        }
    }
}
