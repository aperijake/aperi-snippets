cmake_minimum_required(VERSION 3.12)
project(kokkos_basic)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

# Set the project languages
enable_language(CXX)

IF(GPU)
  enable_language(CUDA)
  set(CUDA_SEPARABLE_COMPILATION ON)
ENDIF()

# Set the path to project-specific custom modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# Find packages
find_package(Kokkos REQUIRED PATHS ${Kokkos_ROOT})

# kokkos_basic
file(GLOB MAIN_SOURCES "src/*.cpp")
add_executable(kokkos_basic ${MAIN_SOURCES})

target_include_directories(kokkos_basic PRIVATE
    "${CMAKE_SOURCE_DIR}/include/"
    ${Kokkos_INCLUDE_DIRS}
)

target_link_libraries(kokkos_basic
    Kokkos::kokkos
)

# Check the build type and set different executable names
set_target_properties(kokkos_basic PROPERTIES
    OUTPUT_NAME "kokkos_basic$<$<CONFIG:Debug>:_debug>$<$<CONFIG:RelWithDebInfo>:_relwithdebinfo>"
)
