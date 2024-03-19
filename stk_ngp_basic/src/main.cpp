// #include <mpi.h>
// #include <iostream>
//
// int main(int argc, char* argv[]) {
//     // Initialize MPI and get communicator for the current process
//     MPI_Init(&argc, &argv);
//     MPI_Comm comm = MPI_COMM_WORLD;
//
//     // Get size of the current process
//     int size;
//     MPI_Comm_size(comm, &size);
//
//     // Print number of processes
//     std::cout << "Running on " << size << " processes." << std::endl;
//
//     // Get input filename from command-line argument
//     std::string input_filename = argv[1];
//
//     // Run the application
//     // RunApplication(input_filename, comm);
//
//     std::cout << "stk ngp hello world example finished successfully!" << std::endl;
//
//     // Finalize MPI and clean up
//     MPI_Finalize();
//
//     return 0;
// }

#include <Kokkos_Core.hpp>
#include <iostream>

int main() {
    // Initialize Kokkos
    Kokkos::initialize();

    // Scope to enforce destruction of Kokkos execution space
    {
        printf("Default Kokkos execution space %s\n", typeid(Kokkos::DefaultExecutionSpace).name());

        // Create a Kokkos parallel for loop that runs on the GPU
        printf("Cuda Kokkos execution space %s\n", typeid(Kokkos::Cuda).name());
        Kokkos::parallel_for(
            "gpu_work", Kokkos::RangePolicy<Kokkos::Cuda>(0, 10), KOKKOS_LAMBDA(int i) {
                printf("Hello from the gpu %d\n", i);
            });

        // Create a Kokkos parallel for loop that runs on the GPU
        printf("Serial Kokkos execution space %s\n", typeid(Kokkos::Serial).name());
        Kokkos::parallel_for(
            "cpu_work", Kokkos::RangePolicy<Kokkos::Serial>(0, 10), KOKKOS_LAMBDA(int i) {
                printf("Hello from the cpu %d\n", i);
            });
    }

    // Finalize Kokkos
    Kokkos::finalize();

    return 0;
}
