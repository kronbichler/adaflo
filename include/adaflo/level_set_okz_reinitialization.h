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

/**
 * Parameters of the reinitialization operator.
 */
struct LevelSetOKZSolverReinitializationParameter
{
  /**
   * TODO
   */
  unsigned int dof_index_ls;

  /**
   * TODO
   */
  unsigned int dof_index_normal;

  /**
   * TODO
   */
  unsigned int quad_index;

  /**
   * TODO
   */
  bool do_iteration;

  /**
   * TODO
   */
  TimeSteppingParameters time;
};

template <int dim>
class LevelSetOKZSolverReinitialization
{
public:
  using VectorType      = LinearAlgebra::distributed::Vector<double>;
  using BlockVectorType = LinearAlgebra::distributed::BlockVector<double>;

  LevelSetOKZSolverReinitialization(
    const BlockVectorType &                           normal_vector_field,
    const AlignedVector<VectorizedArray<double>> &    cell_diameters,
    const double &                                    epsilon_used,
    const double &                                    minimal_edge_length,
    const AffineConstraints<double> &                 constraints,
    VectorType &                                      solution_update,
    VectorType &                                      solution,
    VectorType &                                      system_rhs,
    const ConditionalOStream &                        pcout,
    const DiagonalPreconditioner<double> &            preconditioner,
    const std::pair<double, double> &                 last_concentration_range,
    const LevelSetOKZSolverReinitializationParameter &parameters,
    bool &                                            first_reinit_step,
    const MatrixFree<dim, double> &                   matrix_free)
    : parameters(parameters)
    , solution(solution)
    , solution_update(solution_update)
    , system_rhs(system_rhs)
    , normal_vector_field(normal_vector_field)
    , matrix_free(matrix_free)
    , constraints(constraints)
    , cell_diameters(cell_diameters)
    , epsilon_used(epsilon_used)
    , minimal_edge_length(minimal_edge_length)
    , last_concentration_range(last_concentration_range)
    , first_reinit_step(first_reinit_step)
    , pcout(pcout)
    , time_stepping(parameters.time)
    , preconditioner(preconditioner)
  {}

  // performs reinitialization
  void
  reinitialize(
    const double                     dt,
    const unsigned int               stab_steps,
    const unsigned int               diff_steps     = 0,
    const std::function<void(bool)> &compute_normal = [](const bool) {});

  void
  reinitialization_vmult(VectorType &      dst,
                         const VectorType &src,
                         const bool        diffuse_only) const;

private:
  template <int ls_degree, bool diffuse_only>
  void
  local_reinitialize(const MatrixFree<dim, double> &              data,
                     VectorType &                                 dst,
                     const VectorType &                           src,
                     const std::pair<unsigned int, unsigned int> &cell_range) const;

  template <int ls_degree, bool diffuse_only>
  void
  local_reinitialize_rhs(const MatrixFree<dim, double> &              data,
                         VectorType &                                 dst,
                         const VectorType &                           src,
                         const std::pair<unsigned int, unsigned int> &cell_range);

  /**
   * Parameters
   */
  const LevelSetOKZSolverReinitializationParameter parameters;

  /**
   * Vector section
   */
  VectorType &solution;        // [o]
  VectorType &solution_update; // [-]
  VectorType &system_rhs;      // [-]

  const BlockVectorType &normal_vector_field; // [i];

  /**
   * MatrixFree
   */
  const MatrixFree<dim> &          matrix_free; // [i]
  const AffineConstraints<double> &constraints; // [i]

  const AlignedVector<VectorizedArray<double>> &         cell_diameters;           // [i]
  const double &                                         epsilon_used;             // [i]
  const double &                                         minimal_edge_length;      // [i]
  const std::pair<double, double> &                      last_concentration_range; // [i]
  bool &                                                 first_reinit_step;        // [?]
  AlignedVector<Tensor<1, dim, VectorizedArray<double>>> evaluated_normal;         // [-]

  /**
   * Utility
   */
  const ConditionalOStream &pcout;         // [i]
  TimeStepping              time_stepping; // [-]

  /**
   * Solver section
   */
  const DiagonalPreconditioner<double> &preconditioner; // [i]
};

#endif
