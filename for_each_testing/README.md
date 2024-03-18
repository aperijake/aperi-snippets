# aperi-snippets

Testing of node and element loop abstractions

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
spack env create aperi-snippets
spacktivate aperi-snippets
```

### Add and Install Required Packages

```bash
# If needed, specify a specific compiler. For example, add `%gcc@11.4.0` at the end of the `spack add` commands
# Add Trilinos and eigen
spack add trilinos +boost +exodus +gtest +hdf5 +stk +zoltan +zoltan2
spack add eigen

# Install Packages
spack install
```

---

### Notes on Specific Installs

#### Ubuntu 22.04, x86_64

Successfully installed on an Ubuntu 22.04 system using `apt-get` to install prerequisites:

- `gcc@11.4.0`
- `cmake@3.22.1`

#### Ubuntu 23.10, x86_64

Successfully installed on an Ubuntu 23.10 system using `apt-get` to install prerequisites:

- `gcc@13.2.0`
- `cmake@3.27.4`

#### M1 Mac

Successfully installed on a M1 Mac with the following:

- `gcc@13.2.0` from Homebrew
- `apple-clang@15.0.0` from Xcode
- `cmake@3.26.4` from Conda

For code coverage, be sure `gcov` is using from the same version of gcc. I had to:

```bash
ln -s /opt/homebrew/bin/gcov-13 /opt/homebrew/bin/gcov
```

Also, had some trouble with `openblas` and `gcc`. Compiled `openblas` with `apple-clang`. The commands are:

```bash
# Add Trilinos and eigen
spack add trilinos +boost +exodus +gtest +hdf5 +stk +zoltan +zoltan2 %gcc ^openblas%apple-clang
spack add eigen %gcc@13.2.0

# Install Packages
spack install
```
