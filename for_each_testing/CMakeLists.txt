cmake_minimum_required(VERSION 3.10)

# Needed to stop shared library warnings.  If Trilinos
# was compiled with shared libraries and cmake thinks the system
# only works with static libraries then there could be an issue
SET_PROPERTY(GLOBAL PROPERTY TARGET_SUPPORTS_SHARED_LIBS TRUE)

# Set the path to project-specific custom modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

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
ENDIF()

IF(NOT CMAKE_C_COMPILER)
	MESSAGE("Setting C compiler.")
SET(CMAKE_C_COMPILER ${Trilinos_C_COMPILER})
ENDIF()

project(for_each_testing)

### EIGEN ###
MESSAGE("   EIGEN_PATH = ${EIGEN_PATH}")
find_package(Eigen3 REQUIRED PATHS ${EIGEN_PATH}/lib/cmake/eigen3 ${EIGEN_PATH} NO_DEFAULT_PATH)

IF(Eigen3_FOUND)
    MESSAGE("   Eigen3_DIR = ${Eigen3_DIR}")
    MESSAGE("    Eigen3_INCLUDE_DIR = ${EIGEN3_INCLUDE_DIR}")
    MESSAGE("    Eigen3_LIBRARIES = ${EIGEN3_LIBRARIES}")
ELSE()
    MESSAGE(FATAL_ERROR "Could not find Eigen3!")
ENDIF()

### for_each_testing ###
file(GLOB LIB_SOURCES "src/*.cpp")

add_library(libfor_each_testing ${LIB_SOURCES})

set_property( TARGET libfor_each_testing PROPERTY CXX_STANDARD 11 )
set_property( TARGET libfor_each_testing PROPERTY CXX_STANDARD_REQUIRED ON )

target_link_libraries(libfor_each_testing
            Eigen3::Eigen
			${Trilinos_LIBRARIES}
			${Trilinos_TPL_LIBRARIES}
            ${EXTRA_LIBS}
			)

set(MAIN_SOURCES
    src/main.cpp;
   )

include_directories("${CMAKE_SOURCE_DIR}/include/")
include_directories(${Trilinos_TPL_INCLUDE_DIRS})
include_directories(${Trilinos_INCLUDE_DIRS})

link_directories(${Trilinos_LIBRARY_DIRS})
link_directories(${Trilinos_TPL_LIBRARY_DIRS})

add_executable(for_each_testing ${MAIN_SOURCES})
target_link_libraries( for_each_testing libfor_each_testing )

set_property( TARGET for_each_testing PROPERTY CXX_STANDARD 11 )
set_property( TARGET for_each_testing PROPERTY CXX_STANDARD_REQUIRED ON )

# Check the build type and set different executable names
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set_target_properties(for_each_testing PROPERTIES OUTPUT_NAME "for_each_testing")
    set_target_properties(libfor_each_testing PROPERTIES OUTPUT_NAME "libfor_each_testing")
elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set_target_properties(for_each_testing PROPERTIES OUTPUT_NAME "for_each_testing_debug")
    set_target_properties(libfor_each_testing PROPERTIES OUTPUT_NAME "libfor_each_testing_debug")
elseif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    set_target_properties(for_each_testing PROPERTIES OUTPUT_NAME "for_each_testing_relwithdebinfo")
    set_target_properties(libfor_each_testing PROPERTIES OUTPUT_NAME "libfor_each_testing_relwithdebinfo")
endif()
