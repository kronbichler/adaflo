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


#ifndef __adaflo_level_set_compute_normal_h
#define __adaflo_level_set_compute_normal_h

#include <deal.II/lac/la_parallel_block_vector.h>

#include <deal.II/matrix_free/matrix_free.h>

#include <adaflo/block_matrix_extension.h>
#include <adaflo/diagonal_preconditioner.h>
#include <adaflo/navier_stokes.h>


using namespace dealii;

template <int dim>
class LevelSetOKZSolverComputeNormal
{
public:
  LevelSetOKZSolverComputeNormal(
    const AlignedVector<VectorizedArray<double>> &   cell_diameters,
    const double &                                   epsilon_used,
    const double &                                   minimal_edge_length,
    const AffineConstraints<double> &                constraints_normals,
    LinearAlgebra::distributed::BlockVector<double> &normal_vector_field,
    const std::shared_ptr<TimerOutput> &             timer,
    const NavierStokes<dim> &                        navier_stokes,
    const FlowParameters &                           parameters,
    const MatrixFree<dim> &                          matrix_free,
    LinearAlgebra::distributed::BlockVector<double> &solution,
    LinearAlgebra::distributed::BlockVector<double> &normal_vector_rhs,
    const DiagonalPreconditioner<double> &           preconditioner,
    const std::shared_ptr<BlockMatrixExtension> &    projection_matrix,
    const std::shared_ptr<BlockILUExtension> &       ilu_projection_matrix)
    : cell_diameters(cell_diameters)
    , epsilon_used(epsilon_used)
    , minimal_edge_length(minimal_edge_length)
    , constraints_normals(constraints_normals)
    , normal_vector_field(normal_vector_field)
    , timer(timer)
    , navier_stokes(navier_stokes)
    , parameters(parameters)
    , matrix_free(matrix_free)
    , solution(solution)
    , normal_vector_rhs(normal_vector_rhs)
    , preconditioner(preconditioner)
    , projection_matrix(projection_matrix)
    , ilu_projection_matrix(ilu_projection_matrix)
  {}

  virtual void
  compute_normal(const bool fast_computation);

  void
  compute_normal_vmult(LinearAlgebra::distributed::BlockVector<double> &      dst,
                       const LinearAlgebra::distributed::BlockVector<double> &sr) const;

private:
  template <int ls_degree, typename Number>
  void
  local_compute_normal(const MatrixFree<dim, Number> &                        data,
                       LinearAlgebra::distributed::BlockVector<Number> &      dst,
                       const LinearAlgebra::distributed::BlockVector<Number> &src,
                       const std::pair<unsigned int, unsigned int> &cell_range) const;
  template <int ls_degree>
  void
  local_compute_normal_rhs(const MatrixFree<dim, double> &                   data,
                           LinearAlgebra::distributed::BlockVector<double> & dst,
                           const LinearAlgebra::distributed::Vector<double> &src,
                           const std::pair<unsigned int, unsigned int> &cell_range) const;


  const AlignedVector<VectorizedArray<double>> &   cell_diameters;
  const double &                                   epsilon_used;
  const double &                                   minimal_edge_length;
  const AffineConstraints<double> &                constraints_normals;
  LinearAlgebra::distributed::BlockVector<double> &normal_vector_field;

  const std::shared_ptr<TimerOutput> &             timer;
  const NavierStokes<dim> &                        navier_stokes;
  const FlowParameters &                           parameters;
  const MatrixFree<dim> &                          matrix_free;
  LinearAlgebra::distributed::BlockVector<double> &solution;
  LinearAlgebra::distributed::BlockVector<double> &normal_vector_rhs;

  const DiagonalPreconditioner<double> &       preconditioner;
  const std::shared_ptr<BlockMatrixExtension> &projection_matrix;
  const std::shared_ptr<BlockILUExtension> &   ilu_projection_matrix;
};

#endif
