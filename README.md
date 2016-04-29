 adaflo: An adaptive finite element flow solver
===============================================

adaflo is an adaptive finite element solver for incompressible fluid flow and
two-phase flow based on the level set method. adaflo is based on the deal.II
finite library, github.com/dealii/dealii, and makes use of advanced
technologies such as parallel adaptive mesh refinement, fast integration based
on sum factorization, and state-of-the-art preconditioning techniques.


#Getting started

### Prerequisites

To use adaflo, a standard development environment with a relatively recent C++
compiler, MPI, and cmake is assumed. Furthermore, the following external
software packages are needed:

* deal.II, using at least version 8.4.0, see www.dealii.org. deal.II must be
  configured to also include the following external packages (no direct access
  to this packages is necessary, except for the interface through deal.II):

* p4est for providing parallel adaptive mesh management on forests of
  quad-trees (2D) or oct-trees (3D). For obtaining p4est, see
  http://www.p4est.org. p4est of at least version 0.3.4.2 is needed for
  adaflo. Installation of p4est can be done via a script provided by deal.II:
```
/path/to/dealii/doc/external-libs/p4est-setup.sh p4est-1.1.tar.gz /path/to/p4est/install
```
  (the last argument specifies the desired installation directory for p4est,
  e.g. $HOME/sw/p4est).

* Trilinos for overlapping Schwarz preconditioners (ILU) and algebraic
  multigrid (ML). For obtaining Trilinos, see http://www.trilinos.org. adaflo
  has been tested against several Trilinos versions. All versions between 11.4
  and 12.6 that work together with deal.II should work with adaflo. This is
  because adaflo uses the stable Epetra stack. For options regarding the
  installation of Trilinos, see the respective instructions at the deal.II
  homepage: https://dealii.org/developer/external-libs/trilinos.html

Given these dependencies, the configuration of deal.II can be done
through the following script:
```
cmake \
    -D CMAKE_CXX_FLAGS="-march=native" \
    -D CMAKE_INSTALL_PREFIX="/path/to/dealii/install/" \
    -D DEAL_II_WITH_MPI="ON" \
    -D DEAL_II_WITH_LAPACK="ON" \
    -D DEAL_II_WITH_P4EST="ON" \
    -D P4EST_DIR="/path/to/p4est/install/" \
    -D DEAL_II_WITH_TRILINOS="ON" \
    -D TRILINOS_DIR="/path/to/trilinos/install" \
    ../deal.II
```

Since the algorithms in adaflo make intensive use of advanced processor
instruction sets (e.g. vectorization through AVX or similar), it is
recommended to enable processor-specific optimizations either manually (second
line, `-march=native`), or through the deal.II configuration option `-D
DEAL_II_ALLOW_PLATFORM_INTROSPECTION="ON"`. The path on the third line
specifies the desired installation directory of deal.II, and the last line
points to the location of the source code of deal.II relative to the folder
where the cmake script is run. After configuration, run

```
make -j8
make install
```

to compile deal.II and install it in the given directory. After installation,
the deal.II source and build folder are no longer necessary (unless you find
bugs in deal.II and need to modify that code). Note that it is also possible
to build adaflo against a build folder of deal.II.


### Configuration of adaflo

The adaflo configuration makes use of scripts from the deal.II library. For
setting up adaflo, it is usually enough to run the two commands in the top
level directory of adaflo:

```
cmake -D DEAL_II_DIR=/path/to/dealii/install .
make -j8
```

#Design of adaflo

adaflo is based on core functionality in the folders `include/adaflo` and
`source`. It contains an interface to parameter files, an incompressible
Navier-Stokes solver, a level set two-phase flow solver based on
the conservative level set method by Olsson, Kreiss and Zahedi, and a phase
field two-phase flow solver. This core functionality is collected in a library
`libadaflo.so` (or `libadaflo.dylib`) against which actual applications can be
linked.

In addition, a set of tests are include in the subfolder `tests/`. These
currently include single-fluid tests of a Beltrami(3D)/Taylor(2D) flow and
Poiseuille flow, and two-phase flow tests for a rising bubble and spurious
currents. While these are fully functional cases and can be used as a basis
for studying new problem cases, they also serve as unit tests for ensuring
proper functionality of adaflo.

Finally, somewhat larger configurations are included in the `applications`
subfolder.

### Setting up a new problem in adaflo

Problems in adaflo are controlled on two levels:

* A user-written C++ file that specifies the computational domain (grid) and
  boundary conditions. This gives the user control over the (limited) deal.II
  mesh generation capabilities, or alternatively allows for reading meshes
  from mesh generators such as the ucd format created by Cubit. In addition,
  curved manifolds can be set this way to make the flow solver and grid
  refinement align along these curves.

* An input file with parameters for the fluids (density, viscosity), settings
  of the mesh (number of adaptive mesh levels), the time stepping, and solver
  settings (solver strategy, number of iterations).

New problems are typically set up by taking one of the provided examples in
the `applications` or `tests` folders and modifying as necessary.
