# stk_ngp_basic

Basic "hello world" testing for `stk` on the GPU

## Building with Spack

### Install Spack

Follow the instructions at [https://spack.readthedocs.io/en/latest/getting_started.html](https://spack.readthedocs.io/en/latest/getting_started.html). In short, install the `spack` prerequisites, clone `spack`, and source the script for shell support.

### Find System Tools

```bash
# Add system compilers. e.g. gcc
spack compiler find

# Use external tools. e.g. cmake
spack external find
```

### Create a Spack Environment

```bash
# Create and activate a spack environment for the project
spack env create stk-ngp-basic
spacktivate stk-ngp-basic
```

### Add and Install Required Packages

#### With GPU

```bash
# If needed, specify a specific compiler. For example, add `%gcc@10.5.0` at the end of the `spack add` commands
# Add kokkos and Trilinos, adjust cuda_arch as needed for your GPU device
spack add kokkos +cuda +cuda_lambda +cuda_relocatable_device_code ~cuda_uvm ~shared +wrapper cuda_arch=75 cxxstd=17
spack add trilinos +boost +cuda +cuda_rdc +exodus +gtest +hdf5 ~shared +stk ~uvm +wrapper +zoltan +zoltan2 cuda_arch=75 cxxstd=17
spack add eigen

# Install Packages
spack install
```

#### Without GPU

```bash
# If needed, specify a specific compiler. For example, add `%gcc@10.5.0` at the end of the `spack add` commands
# Add kokkos and Trilinos
spack add kokkos ~cuda ~shared cxxstd=17
spack add trilinos +boost ~cuda +exodus +gtest +hdf5 ~shared +stk +wrapper +zoltan +zoltan2 cxxstd=17
spack add eigen

# Install Packages
spack install
```

---

### Notes on Specific Installs

#### Ubuntu 22.04, x86_64, AWS EC2 g4dn.xlarge

Successfully installed on an AWS EC2 g4dn.xlarge (NVIDIA T4 GPU) Ubuntu 22.04 system using `apt-get` to install prerequisites:

- `gcc@10.5.0`
