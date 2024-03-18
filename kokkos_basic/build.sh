#!/bin/bash

mkdir -p build
cd build
cmake_command="cmake"
cmake_command+=" -D KOKKOS_PATH:FILEPATH=$(spack location -i kokkos)"
cmake_command+=" .."
eval "${cmake_command}"
make
