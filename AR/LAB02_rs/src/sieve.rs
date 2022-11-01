pub type Procs = usize;

pub const BLOCK_FIRST: Procs = 3;
pub const BLOCK_STEP: Procs = 2;


pub fn block_low(id: Procs, p: Procs, n: Procs) -> Procs {
    (id) * (n) / (p) as Procs / BLOCK_STEP
}

pub fn block_high(id: Procs, p: Procs, n: Procs) -> Procs {
    block_low((id) + 1, p, n) - 1
}

pub fn block_size(id: Procs, p: Procs, n: Procs) -> Procs {
    block_low(id + 1, p, n) - block_low(id, p, n)
}

pub fn setup_primes_arr(size: Procs) -> Vec<u8> {

    let mut primes = vec![0 as u8; size];

    (2..size).step_by(2)
        .for_each(|idx| primes[idx] = 1);

    (3..size).step_by(2)
        .for_each(|idx| {
            if primes[idx] != 1 {
                let mut prime_multiple = idx << 1;
                while prime_multiple < size {
                    primes[prime_multiple] = 1;
                    prime_multiple += idx;
                }
            }
        });
    primes
}