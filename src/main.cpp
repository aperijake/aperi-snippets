#include <mpi.h>

#include <iostream>

#include "Benchmarking.h"

void RunTest() {
    // Run the benchmarking test
    BenchmarkingTest();
}

int main(int argc, char* argv[]) {
    // Initialize MPI and get communicator for the current process
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;

    // Get size of the current process
    int size;
    int rank;
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);

    // Print number of processes
    if (rank == 0) {
        std::cout << "Running on " << size << " processes." << std::endl;
    }

    // Run the application
    RunTest();

    if (rank == 0)
    {
        std::cout << "aperi-snippets finished successfully!" << std::endl;
    }

    // Finalize MPI and clean up
    MPI_Finalize();

    return 0;
}