#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

using std::memcpy;
using std::swap;
using std::vector;


void batcher_odd_even_merge_parallel(int *array, int rank, int size, int n) {
    MPI_Status status;
    int base_chunk_size = n / size;
    int s;

    for (s = size; rank % s != 0; s /= 2);
    int chunk_size = s * base_chunk_size;
    int *chunk = rank == 0 ? array : new int[chunk_size];

    if (rank == 0) {
        for (int i = 1; i < size; i++) {
            MPI_Send(array + i * base_chunk_size, base_chunk_size, MPI_INT, i, 0, MPI_COMM_WORLD);
        }
        memcpy(chunk, array, base_chunk_size * sizeof(int));
    } else {
        MPI_Recv(chunk, base_chunk_size, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
    }

    for (int p = 1; p < base_chunk_size; p *= 2) {
        for (int k = p; k >= 1; k /= 2) {
            for (int j = k % p; j < (base_chunk_size - k); j += 2 * k) {
                for (int i = 0; i < k; i++) {
                    if ((i + j) / (p * 2) == (i + j + k) / (p * 2) && chunk[i + j] > chunk[i + j + k]) {
                        swap(chunk[i + j], chunk[i + j + k]);
                    }
                }
            }
        }
    }

    int step = 1;
    while (base_chunk_size < n) {
        if (chunk_size < base_chunk_size * 2) {
            MPI_Send(chunk, chunk_size, MPI_INT, rank - step, 0, MPI_COMM_WORLD);
            break;
        } else {
            MPI_Recv(chunk + base_chunk_size, base_chunk_size, MPI_INT, rank + step, 0, MPI_COMM_WORLD, &status);
            int p = base_chunk_size;
            for (int k = p; k >= 1; k /= 2) {
                for (int j = k % p; j < (base_chunk_size * 2 - k); j += 2 * k) {
                    for (int i = 0; i < k; i++) {
                        if ((i + j) / (p * 2) == (i + j + k) / (p * 2) && chunk[i + j] > chunk[i + j + k]) {
                            swap(chunk[i + j], chunk[i + j + k]);
                        }
                    }
                }
            }
            base_chunk_size *= 2;
            step *= 2;
        }
    }
    if (rank != 0) {
        delete[] chunk;
    }
}


int main(int argc, char **argv) {
    int rank;
    int size;
    double global_result;
    double max_delta_t;

    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (argc != 2) {
        perror("Got invalid args count");
        return 1;
    }
    char *err;
    int array_size = strtol(argv[1], &err, 10);
    if (*err != 0 || array_size < 0) {
        perror("Got invalid array size");
        return 1;
    }

    vector<int> array;
    if (rank == 0) {
        for (int i = 0; i < array_size; i++) {
            array.push_back(rand() % 10000);
        }
    }

    double start = MPI_Wtime();

    batcher_odd_even_merge_parallel(&(array[0]), rank, size, array_size);

    double delta_t = MPI_Wtime() - start;

    if (rank == 0) {
        printf("%d %d %.2f\n", array_size, size, delta_t * 1000);
    }

    MPI_Finalize();

    return 0;
}
