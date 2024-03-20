#include <mpi.h>

#include <Kokkos_Core.hpp>
#include <iostream>

void HelloWorld() {
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
}

int main(int argc, char* argv[]) {
    // Initialize Kokkos and MPI
    Kokkos::initialize();
    MPI_Init(&argc, &argv);

    MPI_Comm comm = MPI_COMM_WORLD;

    // Get size of the current process
    int size;
    MPI_Comm_size(comm, &size);

    // Print number of processes
    std::cout << "Running on " << size << " processes." << std::endl;

    // Run the application
    HelloWorld();

    std::cout << "stk ngp hello world example finished successfully!" << std::endl;

    // Finalize Kokkos and MPI
    Kokkos::finalize();
    MPI_Finalize();

    return 0;
}
