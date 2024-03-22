cmake_minimum_required(VERSION 3.20)
project(stk_ngp_basic)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)

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
find_package(Trilinos REQUIRED PATHS ${TRILINOS_PATH}/lib/cmake/Trilinos ${TRILINOS_PATH})
find_package(MPI REQUIRED)
find_package(Eigen3 REQUIRED PATHS ${EIGEN_PATH}/lib/cmake/eigen3 ${EIGEN_PATH})

# stk_ngp_basic
file(GLOB MAIN_SOURCES "src/*.cpp")
add_executable(stk_ngp_basic ${MAIN_SOURCES})

target_include_directories(stk_ngp_basic PRIVATE
    "${CMAKE_SOURCE_DIR}/include/"
    ${Trilinos_TPL_INCLUDE_DIRS}
    ${Trilinos_INCLUDE_DIRS}
    ${Kokkos_INCLUDE_DIRS}
)

target_link_libraries(stk_ngp_basic
    Kokkos::kokkos
    Eigen3::Eigen
    ${Trilinos_LIBRARIES}
    ${Trilinos_TPL_LIBRARIES}
    ${MPI_LIBRARIES}
)

# Check the build type and set different executable names
set_target_properties(stk_ngp_basic PROPERTIES
    OUTPUT_NAME "stk_ngp_basic$<$<CONFIG:Debug>:_debug>$<$<CONFIG:RelWithDebInfo>:_relwithdebinfo>"
)
