//
// Created by xgg on 27 Oct 2022.
//
#include <stdlib.h>
#include "sieve.h"
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <mpi.h>

int classic(int n) {
    bool arr[n + 1];
    memset(arr, true, sizeof(arr));

    for (int p = 2; p * p <= n; p++) {
        if (arr[p] == true) {
            for (int i = p * p; i <= n; i += p)
                arr[i] = false;
        }
    }

    for (int p = 2; p <= n; p++)
        if (arr[p])
            printf("%d ", p);

    return 0;
}

int prepare_primes(bool *arr, size_t len) {
    for (int p = 2; p * p <= len; p++) {
        if (arr[p] == true) {
            for (int i = p * p; i <= len; i += p)
                arr[i] = false;
        }
    }
    return 0;
}

int sieve(uint low_value, uint high_value, uint size, uint n, int id, int p, uint *res) {
    int count;
    int first;
    int i;
    int prime;
    int first_value_index;
    int prime_step;
    int prime_doubled;
    int sqrt_n;
    int num_per_block;
    int block_low_value;
    int block_high_value;
    int first_index_in_block;
    bool *marked;
    bool *primes;

    sqrt_n = sqrt(n);
    if (NULL == (primes = (bool *) calloc(sqrt_n + 1, sizeof(bool)))) {
        // panic
        exit(10);
    }
    memset(primes, true, sqrt_n + 1);
    prepare_primes(primes, sqrt_n);

    //
    if (NULL == (marked = (bool *) calloc(size * sizeof(bool), sizeof(bool)))) {
        // panic
        exit(11);
    }
    memset(marked, true, size);

    num_per_block = 1024 * 1024;
    block_low_value = low_value;
    block_high_value = MIN(high_value,
                           low_value + num_per_block * BLOCK_STEP);

    for (first_index_in_block = 0;
         first_index_in_block < size;
         first_index_in_block += num_per_block) {
        for (prime = 3; prime <= sqrt_n; ++prime) {
            if (primes[prime] == false)
                continue;
            if (prime * prime > block_low_value) {
                first = prime * prime;
            } else {
                if (!(block_low_value % prime)) {
                    first = block_low_value;
                } else {
                    first = prime - (block_low_value % prime) +
                            block_low_value;
                }
            }
            if ((first + prime) & 1)
                first += prime;

            first_value_index = (first - BLOCK_FIRST) / BLOCK_STEP -
                                BLOCK_LOW(id, p, n - 1);
            prime_doubled = prime << 1;
            prime_step = prime_doubled / BLOCK_STEP;
            for (i = first; i <= high_value; i += prime_doubled) {
                marked[first_value_index] = false;
                first_value_index += prime_step;
            }
        }

        block_low_value += num_per_block * BLOCK_STEP;
        block_high_value = MIN(high_value,
                               block_high_value + num_per_block * BLOCK_STEP);
    }

    count = 0;
    for (i = 0; i < size; i++)
        if (marked[i] == true)
            count++;

    free(primes);
    free(marked);

    *res = count;

    return 0;
}