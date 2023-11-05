# --------------------------------------------------------------------------
#
# Copyright (C) 2013 - 2016 by the adaflo authors
#
# This file is part of the adaflo library.
#
# The adaflo library is free software; you can use it, redistribute it,
# and/or modify it under the terms of the GNU Lesser General Public License
# as published by the Free Software Foundation; either version 2.1 of the
# License, or (at your option) any later version.  The full text of the
# license can be found in the file LICENSE at the top level of the adaflo
# distribution.
#
# --------------------------------------------------------------------------

MESSAGE("====================================================")
MESSAGE("=============== Configuring ADAFLO =================")
MESSAGE("========= An adaptive parallel flow solver =========")
MESSAGE("====================================================")


# this is the standard deal.II search mechanism, including check for Trilinos and p4est

CMAKE_MINIMUM_REQUIRED(VERSION 2.8.12)

FIND_PACKAGE(deal.II 9.3 QUIET
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  )
IF(NOT ${deal.II_FOUND})
  MESSAGE(FATAL_ERROR "\n"
    "*** Could not locate deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
    )
ENDIF()

IF(NOT DEAL_II_WITH_TRILINOS OR NOT DEAL_II_WITH_P4EST)
  MESSAGE(
    "\nadaflo requires a deal.II installation built with support for Trilinos and p4est but one or both of these appears to be missing!\n"
    )
  IF(NOT DEAL_II_WITH_TRILINOS)
    MESSAGE("-- deal.II was built without support for Trilinos!\n")
  ENDIF()

  IF(NOT DEAL_II_WITH_P4EST)
    MESSAGE("-- deal.II was built without support for p4est!\n")
  ENDIF()

  MESSAGE(FATAL_ERROR "")
ENDIF()

DEAL_II_INITIALIZE_CACHED_VARIABLES()

# Set the source files to be compiled
SET( TARGET_SRC
     source/level_set_okz_preconditioner.cc
     source/level_set_okz_compute_curvature.cc
     source/level_set_okz_advance_concentration.cc
     source/level_set_okz_compute_normal.cc
     source/level_set_okz.cc
     source/level_set_okz_reinitialization.cc
     source/phase_field.cc
     source/phase_field_local.cc
     source/navier_stokes_matrix.cc
     source/navier_stokes_preconditioner.cc
     source/navier_stokes.cc
     source/level_set_okz_matrix.cc
     source/parameters.cc
     source/time_stepping.cc
     source/flow_base_algorithm.cc
     source/diagonal_preconditioner.cc
     source/two_phase_base.cc
     source/level_set_base.cc)

# Set the include directory and the name of the project
INCLUDE_DIRECTORIES(include)

PROJECT(adaflo)

ADD_LIBRARY(adaflo  ${TARGET_SRC})


# Define custom targets to easily switch the build type:
ADD_CUSTOM_TARGET(debug
  COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Debug ${CMAKE_SOURCE_DIR}
  COMMENT "Switch CMAKE_BUILD_TYPE to Debug, now type 'make' to build"
  )

ADD_CUSTOM_TARGET(release
  COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Release ${CMAKE_SOURCE_DIR}
  COMMENT "Switch CMAKE_BUILD_TYPE to Release, now type 'make' to build"
  )

DEAL_II_INITIALIZE_CACHED_VARIABLES()

DEAL_II_SETUP_TARGET(adaflo)

ADD_SUBDIRECTORY(applications)

# Set up unit tests
IF(EXISTS ${CMAKE_SOURCE_DIR}/tests/CMakeLists.txt)
  ENABLE_TESTING()
  ADD_SUBDIRECTORY(tests)
ENDIF()


MESSAGE("====================================================")
