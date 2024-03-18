cmake_minimum_required(VERSION 3.12)
project(kokkos_basic LANGUAGES CXX CUDA)

MESSAGE("   KOKKOS_PATH = ${KOKKOS_PATH}")

# Find Kokkos
find_package(Kokkos PATHS ${KOKKOS_PATH} REQUIRED)

# Add executable
add_executable(kokkos_basic main.cpp)

# Include Kokkos headers
target_include_directories(kokkos_basic PRIVATE ${Kokkos_INCLUDE_DIRS})

# Link against Kokkos
target_link_libraries(kokkos_basic PRIVATE Kokkos::kokkos)

# Set CUDA as the language for this project
# enable_language(CUDA)

# Set the CUDA architecture (adjust based on your GPU)
target_compile_options(kokkos_basic PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_75>)

# Set the CUDA standard to 14, compatible with Kokkos
set_property(TARGET kokkos_basic PROPERTY CUDA_STANDARD 14)
