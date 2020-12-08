// --------------------------------------------------------------------------
//
// Copyright (C) 2008 - 2016 by the adaflo authors
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


#ifndef __adaflo_level_set_reinitialization_h
#define __adaflo_level_set_reinitialization_h

#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/matrix_free/matrix_free.h>

#include <adaflo/diagonal_preconditioner.h>
#include <adaflo/level_set_okz_compute_normal.h>
#include <adaflo/navier_stokes.h>
#include <adaflo/parameters.h>
#include <adaflo/time_stepping.h>

using namespace dealii;

template <int dim>
class LevelSetOKZSolverReinitialization
{
public:
  LevelSetOKZSolverReinitialization(
    LevelSetOKZSolverComputeNormal<dim> &                   normal_operator,
    const LinearAlgebra::distributed::BlockVector<double> & normal_vector_field,
    const AlignedVector<VectorizedArray<double>> &          cell_diameters,
    const double &                                          epsilon_used,
    const double &                                          minimal_edge_length,
    const AffineConstraints<double> &                       constraints,
    LinearAlgebra::distributed::BlockVector<double> &       solution_update,
    LinearAlgebra::distributed::BlockVector<double> &       solution,
    LinearAlgebra::distributed::BlockVector<double> &       system_rhs,
    const ConditionalOStream &                              pcout,
    const DiagonalPreconditioner<double> &                  preconditioner,
    const std::pair<double, double> &                       last_concentration_range,
    const FlowParameters &                                  parameters,
    const TimeStepping &                                    time_stepping,
    bool &                                                  first_reinit_step,
    const MatrixFree<dim, double> &                         matrix_free,
    AlignedVector<Tensor<1, dim, VectorizedArray<double>>> &evaluated_convection)
    : normal_operator(normal_operator)
    , normal_vector_field(normal_vector_field)
    , cell_diameters(cell_diameters)
    , epsilon_used(epsilon_used)
    , minimal_edge_length(minimal_edge_length)
    , constraints(constraints)
    , solution_update(solution_update)
    , solution(solution)
    , system_rhs(system_rhs)
    , pcout(pcout)
    , preconditioner(preconditioner)
    , last_concentration_range(last_concentration_range)
    , parameters(parameters)
    , time_stepping(time_stepping)
    , first_reinit_step(first_reinit_step)
    , matrix_free(matrix_free)
    , evaluated_convection(evaluated_convection)
  {}

  // performs reinitialization
  virtual void
  reinitialize(const unsigned int stab_steps,
               const unsigned int diff_steps                              = 0,
               const bool         diffuse_cells_with_large_curvature_only = false);

  void
  reinitialization_vmult(LinearAlgebra::distributed::Vector<double> &      dst,
                         const LinearAlgebra::distributed::Vector<double> &src,
                         const bool diffuse_only) const;

private:
  template <int ls_degree, bool diffuse_only>
  void
  local_reinitialize(const MatrixFree<dim, double> &                   data,
                     LinearAlgebra::distributed::Vector<double> &      dst,
                     const LinearAlgebra::distributed::Vector<double> &src,
                     const std::pair<unsigned int, unsigned int> &     cell_range) const;

  template <int ls_degree, bool diffuse_only>
  void
  local_reinitialize_rhs(const MatrixFree<dim, double> &                   data,
                         LinearAlgebra::distributed::Vector<double> &      dst,
                         const LinearAlgebra::distributed::Vector<double> &src,
                         const std::pair<unsigned int, unsigned int> &     cell_range);

  LevelSetOKZSolverComputeNormal<dim> &normal_operator;

  const LinearAlgebra::distributed::BlockVector<double> &normal_vector_field;

  const AlignedVector<VectorizedArray<double>> &cell_diameters;

  const double &epsilon_used;
  const double &minimal_edge_length;

  const AffineConstraints<double> &constraints;

  LinearAlgebra::distributed::BlockVector<double> &solution_update;
  LinearAlgebra::distributed::BlockVector<double> &solution;
  LinearAlgebra::distributed::BlockVector<double> &system_rhs;

  const ConditionalOStream &                              pcout;
  const DiagonalPreconditioner<double> &                  preconditioner;
  const std::pair<double, double> &                       last_concentration_range;
  const FlowParameters &                                  parameters;
  const TimeStepping &                                    time_stepping;
  bool &                                                  first_reinit_step;
  const MatrixFree<dim> &                                 matrix_free;
  AlignedVector<Tensor<1, dim, VectorizedArray<double>>> &evaluated_convection;
};

#endif
