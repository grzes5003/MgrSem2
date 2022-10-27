//
// Created by xgg on 27 Oct 2022.
//
#include <stdlib.h>
#include "sieve.h"
#include <math.h>
//#include <stdbool.h>
#include <string.h>
#include <stdio.h>
#include <mpi.h>

int get_vals(uint start_value, uint len, uint n) {
    int id;
    int proc_num;

    uint global_count;

    uint sqrt_n;
    uint i;
    uint prime;
    uint idx;
    uint first;

    uint num_per_block;
    uint block_low_value;
    uint block_high_value;

    uint low_value;
    uint high_value;
    uint size;

    uint prime_doubled;
    uint first_value_index;
    uint prime_step;

    uint first_index_in_block;

    bool *primes;
    bool *marked;

    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &proc_num);

//    const uint end_value = start_value + len;
    sqrt_n = (uint) sqrt(n);

    low_value = BLOCK_FIRST + BLOCK_LOW(id, proc_num, n - 1) * BLOCK_STEP;
    high_value = BLOCK_FIRST + BLOCK_HIGH(id, proc_num, n - 1) * BLOCK_STEP;
    size = BLOCK_SIZE(id, proc_num, n - 1);



    //
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


    //
    num_per_block = 1024 * 1024;
    block_low_value = low_value;
    block_high_value = MIN(high_value, low_value + num_per_block * BLOCK_STEP);

    for (first_index_in_block = 0;
         first_index_in_block < size;
         first_index_in_block += num_per_block) {
        for (prime = 3; prime <= sqrt_n; prime++) {
            if (primes[prime] == 1)
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

            /*
             * optimization - consider only odd multiples
             *                of the prime number
             */
            if ((first + prime) & 1) // is odd
                first += prime;

            first_value_index = (first - BLOCK_FIRST) / BLOCK_STEP -
                                BLOCK_LOW(id, proc_num, n - 1);
            prime_doubled = prime << 1;
            prime_step = prime_doubled / BLOCK_STEP;
            for (i = first; i <= high_value; i += prime_doubled) {
                marked[first_value_index] = 1;
                first_value_index += prime_step;
            } /* for */
        }

        block_low_value += num_per_block * BLOCK_STEP;
        block_high_value = MIN(high_value,
                               block_high_value + num_per_block * BLOCK_STEP);
    } /* for first_index_in_block */

    uint count = 0;
    for (i = 0; i < size; i++)
        if (marked[i] == false) {
            printf(";%d ", i);
            count++;
        }
    printf("\n");

    MPI_Reduce(&count, &global_count, 1, MPI_INT,
               MPI_SUM, 0, MPI_COMM_WORLD);


    /* print the results */
    printf("I am %d\n", id);
    if (id == 0) {
        global_count += 1; /* add first prime, 2 */
        printf("%d primes are less than or equal to %d\n",
               global_count, n);
    } /* if */


    MPI_Finalize();

    free(primes);
    free(marked);

    return 0;
}

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


int prime(int n)		{
    int     count;                /* local prime count */
    double  elapsed_time;         /* parallel execution time */
    int     first;                /* index of first multiple */
    int     global_count;         /* global prime count */
    int     high_value;           /* highest value on this proc */
    int     i;
    int     id;                   /* process id number */
    int     index;                /* index of current prime */
    int     low_value;            /* lowest value on this proc */
    int     p;                    /* number of processes */
    int     proc0_size;           /* size of proc 0's subarray */
    int     prime;                /* current prime */
    int     size;                 /* elements in marked string */
    int     first_value_index;
    int     prime_step;
    int     prime_doubled;
    int     sqrt_n;
    int     prime_multiple;
    int     num_per_block;
    int     block_low_value;
    int     block_high_value;
    int     first_index_in_block;
    char*   marked;               /* portion of 2, ..., n */
    char*   primes;


    /*
     * bail out if all the primes used for sieving are not all
     * help by process 0
     */
    proc0_size = (n - 1) / p;

    if ((2 + proc0_size) < (int)sqrt((double)n))    {
        if (id == 0) /* parent process */
            printf("Too many processes\n");
        MPI_Finalize();
        exit(1);
    } /* if */

    // compute primes from 2 to sqrt(n);
    sqrt_n = sqrt(n);
    primes = (char*)calloc(sqrt_n + 1, 1);
    for (prime_multiple = 2;
         prime_multiple <= sqrt_n;
         prime_multiple += 2)    {
        primes[prime_multiple] = 1;
    } /* for */

    for (prime = 3; prime <= sqrt_n; prime += 2)    {
        if (primes[prime] == 1)
            continue;

        for (prime_multiple = prime << 1;
             prime_multiple <= sqrt_n;
             prime_multiple += prime)    {
            primes[prime_multiple] = 1;
        }
    } /* for */

    /*
     * allocate this process' share of the array
     */
    marked = (char*)calloc(size * sizeof(char), 1);
    if (marked == NULL)    {
        printf("Cannot allocate enough memory\n");
        MPI_Finalize();
        exit(1);
    } /* if */

    num_per_block    = 1024 * 1024;
    block_low_value  = low_value;
    block_high_value = MIN(high_value,
                           low_value + num_per_block * BLOCK_STEP);

    for (first_index_in_block = 0;
         first_index_in_block < size;
         first_index_in_block += num_per_block)    {
        for (prime = 3; prime <= sqrt_n; prime++)       {
            if (primes[prime] == 1)
                continue;
            if (prime * prime > block_low_value)   {
                first = prime * prime;
            }
            else   {
                if (!(block_low_value % prime))    {
                    first = block_low_value;
                }
                else    {
                    first = prime - (block_low_value % prime) +
                            block_low_value;
                }
            }

            /*
             * optimization - consider only odd multiples
             *                of the prime number
             */
            if ((first + prime) & 1) // is odd
                first += prime;

            first_value_index = (first - BLOCK_FIRST) / BLOCK_STEP -
                                BLOCK_LOW(id, p, n - 1);
            prime_doubled     = prime << 1;
            prime_step        = prime_doubled / BLOCK_STEP;
            for (i = first; i <= high_value; i += prime_doubled)   {
                marked[first_value_index] = 1;
                first_value_index += prime_step;
            } /* for */
        }

        block_low_value += num_per_block * BLOCK_STEP;
        block_high_value = MIN(high_value,
                               block_high_value + num_per_block * BLOCK_STEP);
    } /* for first_index_in_block */


    /*
     * count the number of prime numbers found on this process
     */
    count = 0;
    for (i = 0; i < size; i++)
        if (!marked[i])
            count++;

    MPI_Reduce(&count, &global_count, 1, MPI_INT,
               MPI_SUM, 0, MPI_COMM_WORLD);

    /*
     * stop the timer
     */
    elapsed_time += MPI_Wtime();

    /* print the results */
    if (id == 0)   {
        global_count += 1; /* add first prime, 2 */
        printf("%d primes are less than or equal to %d\n",
               global_count, n);
        printf("Total elapsed time: %10.6fs\n",
               elapsed_time);
    } /* if */

    MPI_Finalize();

    return 0;
}

int modd(uint low_value, uint high_value, uint size, uint n, int id, int p, uint *res) {
    int     count;                /* local prime count */
    int     first;                /* index of first multiple */
    int     i;
    int     prime;                /* current prime */
    int     first_value_index;
    int     prime_step;
    int     prime_doubled;
    int     sqrt_n;
    int     num_per_block;
    int     block_low_value;
    int     block_high_value;
    int     first_index_in_block;
    bool*   marked;               /* portion of 2, ..., n */
    bool*   primes;

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

    num_per_block    = 1024 * 1024;
    block_low_value  = low_value;
    block_high_value = MIN(high_value,
                           low_value + num_per_block * BLOCK_STEP);

    for (first_index_in_block = 0;
         first_index_in_block < size;
         first_index_in_block += num_per_block)    {
        for (prime = 3; prime <= sqrt_n; ++prime)       {
            if (primes[prime] == false)
                continue;
            if (prime * prime > block_low_value)   {
                first = prime * prime;
            }
            else   {
                if (!(block_low_value % prime))    {
                    first = block_low_value;
                }
                else    {
                    first = prime - (block_low_value % prime) +
                            block_low_value;
                }
            }

            if ((first + prime) & 1) // is odd
                first += prime;

            first_value_index = (first - BLOCK_FIRST) / BLOCK_STEP -
                                BLOCK_LOW(id, p, n - 1);
            prime_doubled     = prime << 1;
            prime_step        = prime_doubled / BLOCK_STEP;
            for (i = first; i <= high_value; i += prime_doubled)   {
                marked[first_value_index] = false;
                first_value_index += prime_step;
            } /* for */
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