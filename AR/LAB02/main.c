#include <mpi.h>
#include <stdio.h>
#include <stdbool.h>
#include <string.h>
#include <stdlib.h>
#include "sieve.h"

int main(int argc, char** argv) {
    double  elapsed_time;

    int id;
    int p;

    uint global_count;

    uint n;
    uint low_value;
    uint high_value;
    uint size;

    uint *result;

    if (argc != 2)    {
        n = 100;
    } else {
        n = atoi(argv[1]);
    }

    // Initialize the MPI environment
    MPI_Init(NULL, NULL);

    /* start the timer */
    MPI_Barrier(MPI_COMM_WORLD);
    elapsed_time = -MPI_Wtime();

    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    low_value  = BLOCK_FIRST + BLOCK_LOW(id, p, n - 1)  * BLOCK_STEP;
    high_value = BLOCK_FIRST + BLOCK_HIGH(id, p, n - 1) * BLOCK_STEP;
    size       = BLOCK_SIZE(id, p, n - 1);

    if (NULL == (result = (uint *) calloc(sizeof(uint), sizeof(uint)))) {
        // panic
        exit(11);
    }
    if (0 != sieve(low_value, high_value, size, n, id, p, result)) {
        printf("sieve exited with unexpected value");
        return 9;
    }

    MPI_Reduce(result, &global_count, 1, MPI_INT,
               MPI_SUM, 0, MPI_COMM_WORLD);

    elapsed_time += MPI_Wtime();

    if (id == 0)   {
        global_count += 1;
        printf("t=%10.6fs;%d;%d\n", elapsed_time, global_count, n);
    }

    MPI_Finalize();

}