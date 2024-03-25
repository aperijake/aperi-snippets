#include <gtest/gtest.h>
#include <mpi.h>

#include <Kokkos_Core.hpp>

int main(int argc, char** argv) {
    Kokkos::initialize();
    MPI_Init(&argc, &argv);

    // Initialize Google Test
    ::testing::InitGoogleTest(&argc, argv);

    MPI_Comm comm = MPI_COMM_WORLD;

    // Get size of the current process
    int size;
    MPI_Comm_size(comm, &size);

    // Print if running on GPU
    Kokkos::DefaultExecutionSpace::concurrency() > 1 ? std::cout << "Running on GPU" << std::endl
                                                     : std::cout << "Running on CPU" << std::endl;

    // Print number of processes
    std::cout << "Running on " << size << " processes." << std::endl;

    // Run all the tests
    int return_code = RUN_ALL_TESTS();

    std::cout << "stk_ngp_basic finished successfully!" << std::endl;

    // Finalize Kokkos and MPI
    Kokkos::finalize();
    MPI_Finalize();

    return return_code;
}