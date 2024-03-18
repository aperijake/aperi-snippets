cmake_minimum_required(VERSION 3.12)
project(KokkosGPUExample LANGUAGES CXX CUDA)

# Find Kokkos
find_package(Kokkos PATHS ${KOKKOS_PATH} REQUIRED)

# Add executable
add_executable(KokkosGPUExample main.cpp)

# Include Kokkos headers
target_include_directories(KokkosGPUExample PRIVATE ${Kokkos_INCLUDE_DIRS})

# Link against Kokkos
target_link_libraries(KokkosGPUExample PRIVATE Kokkos::kokkos)

# Set CUDA as the language for this project
# enable_language(CUDA)

# Set the CUDA architecture (adjust based on your GPU)
target_compile_options(KokkosGPUExample PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_75>)

# Set the CUDA standard to 14, compatible with Kokkos
set_property(TARGET KokkosGPUExample PROPERTY CUDA_STANDARD 14)
