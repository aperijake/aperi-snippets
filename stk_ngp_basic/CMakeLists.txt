cmake_minimum_required(VERSION 3.12)

project(stk_ngp_basic)

#SET_PROPERTY( GLOBAL PROPERTY TARGET_SUPPORTS_SHARED_LIBS TRUE)
set_property( GLOBAL PROPERTY CUDA_STANDARD 17)
set_property( GLOBAL PROPERTY CXX_STANDARD 17 )
set_property( GLOBAL PROPERTY CXX_STANDARD_REQUIRED ON )
IF(GPU)
  set(CUDA_SEPARABLE_COMPILATION ON)
  set(CMAKE_CUDA_FLAGS "-arch=sm_75")
ENDIF()

# Set the path to project-specific custom modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# Set the project languages
enable_language(CXX)
MESSAGE("GPU = ${GPU}")
IF(GPU)
    enable_language(CUDA)
ENDIF()

# Find Kokkos
find_package(Kokkos PATHS ${Kokkos_ROOT} REQUIRED)

### TRILINOS ###
MESSAGE("   TRILINOS_PATH = ${TRILINOS_PATH}")

FIND_PACKAGE(Trilinos PATHS ${TRILINOS_PATH}/lib/cmake/Trilinos ${TRILINOS_PATH} NO_DEFAULT_PATH)

IF(Trilinos_FOUND)
 MESSAGE("\nFound Trilinos!  Here are the details: ")
 MESSAGE("   Trilinos_DIR = ${Trilinos_DIR}")
 MESSAGE("   Trilinos_VERSION = ${Trilinos_VERSION}")
 MESSAGE("   Trilinos_PACKAGE_LIST = ${Trilinos_PACKAGE_LIST}")
 MESSAGE("   Trilinos_LIBRARIES = ${Trilinos_LIBRARIES}")
 MESSAGE("   Trilinos_INCLUDE_DIRS = ${Trilinos_INCLUDE_DIRS}")
 MESSAGE("   Trilinos_TPL_LIST = ${Trilinos_TPL_LIST}")
 MESSAGE("   Trilinos_TPL_INCLUDE_DIRS = ${Trilinos_TPL_INCLUDE_DIRS}")
 MESSAGE("   Trilinos_TPL_LIBRARIES = ${Trilinos_TPL_LIBRARIES}")
 MESSAGE("   Trilinos_BUILD_SHARED_LIBS = ${Trilinos_BUILD_SHARED_LIBS}")
 MESSAGE("   Trilinos_CXX_COMPILER = ${Trilinos_CXX_COMPILER}")
 MESSAGE("   Trilinos_C_COMPILER = ${Trilinos_C_COMPILER}")
 MESSAGE("   Trilinos_CUDA_COMPILER = ${Trilinos_CUDA_COMPILER}")
 MESSAGE("   Trilinos_Fortran_COMPILER = ${Trilinos_Fortran_COMPILER}")
 MESSAGE("   Trilinos_CXX_COMPILER_FLAGS = ${Trilinos_CXX_COMPILER_FLAGS}")
 MESSAGE("   Trilinos_C_COMPILER_FLAGS = ${Trilinos_C_COMPILER_FLAGS}")
 MESSAGE("   Trilinos_Fortran_COMPILER_FLAGS = ${Trilinos_Fortran_COMPILER_FLAGS}")
 MESSAGE("   Trilinos_LINKER = ${Trilinos_LINKER}")
 MESSAGE("   Trilinos_EXTRA_LD_FLAGS = ${Trilinos_EXTRA_LD_FLAGS}")
 MESSAGE("   Trilinos_AR = ${Trilinos_AR}")
 MESSAGE("End of Trilinos details\n")
ELSE()
 MESSAGE(FATAL_ERROR "Could not find Trilinos!")
ENDIF()

IF(NOT CMAKE_CXX_COMPILER)
 MESSAGE("Setting CXX compiler.")
  SET(CMAKE_CXX_COMPILER ${Trilinos_CXX_COMPILER})
  MESSAGE(" CXX compiler: ${CMAKE_CXX_COMPILER}")
ENDIF()

IF(NOT CMAKE_C_COMPILER)
 MESSAGE("Setting C compiler.")
  SET(CMAKE_C_COMPILER ${Trilinos_C_COMPILER})
  MESSAGE(" C compiler: ${CMAKE_C_COMPILER}")
ENDIF()

find_package(MPI REQUIRED)

### stk_ngp_basic ###
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
    ${Trilinos_LIBRARIES}
    ${Trilinos_TPL_LIBRARIES}
    ${EXTRA_LIBS}
    ${MPI_LIBRARIES}
)

# Check the build type and set different executable names
if(CMAKE_BUILD_TYPE STREQUAL "Release")
  set_target_properties(stk_ngp_basic PROPERTIES OUTPUT_NAME "stk_ngp_basic")
elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set_target_properties(stk_ngp_basic PROPERTIES OUTPUT_NAME "stk_ngp_basic_debug")
elseif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
  set_target_properties(stk_ngp_basic PROPERTIES OUTPUT_NAME "stk_ngp_basic_relwithdebinfo")
endif()