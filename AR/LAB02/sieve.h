//
// Created by xgg on 27 Oct 2022.
//

#ifndef LAB02_SIEVE_H
#define LAB02_SIEVE_H

#include <stdbool.h>

typedef unsigned int uint;

#define BLOCK_FIRST 3 /* first odd prime number */
#define BLOCK_STEP 2  /* loop step to iterate only for odd numbers */

#define MIN(a, b) ((a) < (b)? (a): (b))

#define BLOCK_LOW(id, p, n) \
        ((id) * (n) / (p) / BLOCK_STEP)

#define BLOCK_HIGH(id, p, n) \
        (BLOCK_LOW((id) + 1, p, n) - 1)

#define BLOCK_SIZE(id, p, n) \
        (BLOCK_LOW((id) + 1, p, n) - BLOCK_LOW((id), p, n))

int classic(int n);

int prepare_primes(bool *arr, size_t len);

int sieve(uint low_value, uint high_value, uint size, uint n, int id, int p, uint *res);

#endif //LAB02_SIEVE_H
