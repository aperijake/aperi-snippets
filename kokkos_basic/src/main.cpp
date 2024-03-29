#include <Kokkos_Core.hpp>
#include <iostream>

void HelloWorld() {
    // Scope to enforce destruction of Kokkos execution space
    {
        bool is_gpu = Kokkos::DefaultExecutionSpace::concurrency() > 1;

        // Create a Kokkos parallel for loop that runs on the GPU
        if (is_gpu) {
            printf("Cuda Kokkos execution space %s\n", typeid(Kokkos::DefaultExecutionSpace).name());
            Kokkos::parallel_for(
                "gpu_work", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, 10), KOKKOS_LAMBDA(int i) {
                    printf("Hello from the gpu %d\n", i);
                });
        }

        // Create a Kokkos parallel for loop that runs on the CPU
        printf("Serial Kokkos execution space %s\n", typeid(Kokkos::Serial).name());
        Kokkos::parallel_for(
            "cpu_work", Kokkos::RangePolicy<Kokkos::Serial>(0, 10), KOKKOS_LAMBDA(int i) {
                printf("Hello from the cpu %d\n", i);
            });
    }
}

int main(int argc, char* argv[]) {
    // Initialize Kokkos
    Kokkos::initialize();

    // Run the application
    HelloWorld();

    std::cout << "kokkos hello world example finished successfully!" << std::endl;

    // Finalize Kokkos
    Kokkos::finalize();

    return 0;
}
