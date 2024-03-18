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
