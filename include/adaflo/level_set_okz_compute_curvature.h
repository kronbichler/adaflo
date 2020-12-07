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


#ifndef __adaflo_level_compute_curvature_h
#define __adaflo_level_compute_curvature_h

#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/matrix_free/matrix_free.h>

#include <adaflo/block_matrix_extension.h>
#include <adaflo/diagonal_preconditioner.h>
#include <adaflo/level_set_okz_compute_normal.h>
#include <adaflo/navier_stokes.h>
#include <adaflo/time_stepping.h>
#include <adaflo/util.h>

using namespace dealii;

template <int dim>
class LevelSetOKZSolverComputeCurvature
{
public:
  LevelSetOKZSolverComputeCurvature(
    LevelSetOKZSolverComputeNormal<dim> &                  normal_operator,
    const AlignedVector<VectorizedArray<double>> &         cell_diameters,
    const LinearAlgebra::distributed::BlockVector<double> &normal_vector_field,
    const AffineConstraints<double> &                      constraints_curvature,
    const AffineConstraints<double> &                      constraints,
    const double &                                         epsilon_used,
    const std::shared_ptr<TimerOutput> &                   timer,
    LinearAlgebra::distributed::BlockVector<double> &      system_rhs,
    const NavierStokes<dim> &                              navier_stokes,
    const FlowParameters &                                 parameters,
    LinearAlgebra::distributed::BlockVector<double> &      solution,
    const MatrixFree<dim> &                                matrix_free,
    const DiagonalPreconditioner<double> &                 preconditioner,
    std::shared_ptr<BlockMatrixExtension> &                projection_matrix,
    std::shared_ptr<BlockILUExtension> &                   ilu_projection_matrix)
    : normal_operator(normal_operator)
    , cell_diameters(cell_diameters)
    , normal_vector_field(normal_vector_field)
    , constraints_curvature(constraints_curvature)
    , constraints(constraints)
    , epsilon_used(epsilon_used)
    , timer(timer)
    , system_rhs(system_rhs)
    , navier_stokes(navier_stokes)
    , parameters(parameters)
    , solution(solution)
    , matrix_free(matrix_free)
    , preconditioner(preconditioner)
    , projection_matrix(projection_matrix)
    , ilu_projection_matrix(ilu_projection_matrix)
  {}



  //  virtual void
  //  compute_normal(const bool fast_computation)
  //  {
  //    normal_operator.compute_normal(fast_computation);
  //  }

  virtual void
  compute_curvature(const bool diffuse_large_values = false);

  void
  compute_curvature_vmult(LinearAlgebra::distributed::Vector<double> &      dst,
                          const LinearAlgebra::distributed::Vector<double> &srcc,
                          const bool apply_diffusion) const;

private:
  // diffusion_setting: 0: both terms, 1: only mass, 2: only diffusion
  template <int ls_degree, int diffusion_setting>
  void
  local_compute_curvature(const MatrixFree<dim, double> &                   data,
                          LinearAlgebra::distributed::Vector<double> &      dst,
                          const LinearAlgebra::distributed::Vector<double> &src,
                          const std::pair<unsigned int, unsigned int> &cell_range) const;
  template <int ls_degree>
  void
  local_compute_curvature_rhs(
    const MatrixFree<dim, double> &                   data,
    LinearAlgebra::distributed::Vector<double> &      dst,
    const LinearAlgebra::distributed::Vector<double> &src,
    const std::pair<unsigned int, unsigned int> &     cell_range) const;


  LevelSetOKZSolverComputeNormal<dim> &                  normal_operator;
  const AlignedVector<VectorizedArray<double>> &         cell_diameters;
  const LinearAlgebra::distributed::BlockVector<double> &normal_vector_field;
  const AffineConstraints<double> &                      constraints_curvature;
  const AffineConstraints<double> &                      constraints;
  const double &                                         epsilon_used;

  const std::shared_ptr<TimerOutput> &             timer;
  LinearAlgebra::distributed::BlockVector<double> &system_rhs;
  const NavierStokes<dim> &                        navier_stokes;
  const FlowParameters &                           parameters;
  LinearAlgebra::distributed::BlockVector<double> &solution;


  const MatrixFree<dim> &matrix_free;

  const DiagonalPreconditioner<double> & preconditioner;
  std::shared_ptr<BlockMatrixExtension> &projection_matrix;
  std::shared_ptr<BlockILUExtension> &   ilu_projection_matrix;
};

#endif
