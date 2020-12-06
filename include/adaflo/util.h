// --------------------------------------------------------------------------
//
// Copyright (C) 2020 by the adaflo authors
//
// This file is part of the adaflo library.
//
// The adaflo library is free software; you can use it, redistribute it,
// and/or modify it under the terms of the GNU Lesser General Public License
// as published by the Free Software Foundation; either version 2.1 of the
// License, or (at your option) any later version.  The full text of the
// license can be found in the file LICENSE at the top level of the adaflo
// distribution.
//
// --------------------------------------------------------------------------

#ifndef __adaflo_util_h_
#define __adaflo_util_h_

#include <deal.II/distributed/tria.h>

using namespace dealii;

/**
 * Return the communicator of an arbitrary mesh type. Use this function to be
 * able to code that is independent if a serial or a parallel triangulation is
 * used.
 */
template <typename MeshType>
MPI_Comm
get_communicator(const MeshType &mesh)
{
  const auto *tria_parallel = dynamic_cast<
    const parallel::TriangulationBase<MeshType::dimension, MeshType::space_dimension> *>(
    &(mesh.get_triangulation()));

  return tria_parallel != nullptr ? tria_parallel->get_communicator() : MPI_COMM_SELF;
}

/**
 * Return the locally-owned subdomain of an arbitrary mesh type. Use this
 * function to be able to code that is independent if a serial or a parallel
 * triangulation is used.
 */
template <typename MeshType>
unsigned int
locally_owned_subdomain(const MeshType &mesh)
{
  const auto *tria_parallel = dynamic_cast<
    const parallel::TriangulationBase<MeshType::dimension, MeshType::space_dimension> *>(
    &(mesh.get_triangulation()));

  return tria_parallel != nullptr ? tria_parallel->locally_owned_subdomain() : 0;
}

#endif
