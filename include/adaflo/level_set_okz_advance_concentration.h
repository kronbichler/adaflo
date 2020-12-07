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


#ifndef __adaflo_level_set_advance_concentration_h
#define __adaflo_level_set_advance_concentration_h

#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/matrix_free/matrix_free.h>

#include <adaflo/diagonal_preconditioner.h>
#include <adaflo/navier_stokes.h>
#include <adaflo/time_stepping.h>
#include <adaflo/util.h>

using namespace dealii;

/**
 * Parameters of the advection-concentration operator.
 */
struct LevelSetOKZSolverAdvanceConcentrationParameter
{
  /**
   * TODO
   */
  unsigned int dof_index_ls;

  /**
   * TODO
   */
  unsigned int dof_index_vel;

  /**
   * TODO
   */
  unsigned int quad_index;

  /**
   * TODO: needed? this is equivalent to `fe.tensor_degree()+1`?
   */
  unsigned int concentration_subdivisions;

  /**
   * TODO
   */
  bool convection_stabilization;

  /**
   * TODO
   */
  bool do_iteration;

  /**
   * TODO
   */
  double tol_nl_iteration;

  /**
   * TODO
   */
  TimeSteppingParameters time;
};

/**
 * Boundary descriptors of the advection-concentration operator.
 */
template <int dim>
struct LevelSetOKZSolverAdvanceConcentrationBoundaryDescriptor
{
  /**
   * TODO
   */
  std::map<types::boundary_id, std::shared_ptr<Function<dim>>> dirichlet;

  /**
   * TODO
   */
  std::set<types::boundary_id> symmetry;
};

template <int dim>
class LevelSetOKZSolverAdvanceConcentration
{
public:
  using VectorType = LinearAlgebra::distributed::Vector<double>;

  LevelSetOKZSolverAdvanceConcentration(
    VectorType &                                  solution,
    const VectorType &                            solution_old,
    const VectorType &                            solution_old_old,
    VectorType &                                  increment,
    VectorType &                                  rhs,
    const VectorType &                            vel_solution,
    const VectorType &                            vel_solution_old,
    const VectorType &                            vel_solution_old_old,
    const double &                                global_omega_diameter,
    const AlignedVector<VectorizedArray<double>> &cell_diameters,
    const AffineConstraints<double> &             constraints,
    const ConditionalOStream &                    pcout,
    const LevelSetOKZSolverAdvanceConcentrationBoundaryDescriptor<dim> &boundary,
    const MatrixFree<dim> &                                             matrix_free,
    const LevelSetOKZSolverAdvanceConcentrationParameter &              parameters,
    AlignedVector<VectorizedArray<double>> &                artificial_viscosities,
    double &                                                global_max_velocity,
    const DiagonalPreconditioner<double> &                  preconditioner,
    AlignedVector<Tensor<1, dim, VectorizedArray<double>>> &evaluated_convection);

  virtual void
  advance_concentration(const double dt);

  void
  advance_concentration_vmult(VectorType &dst, const VectorType &src) const;

private:
  template <int ls_degree, int velocity_degree>
  void
  local_advance_concentration(
    const MatrixFree<dim, double> &              data,
    VectorType &                                 dst,
    const VectorType &                           src,
    const std::pair<unsigned int, unsigned int> &cell_range) const;

  template <int ls_degree, int velocity_degree>
  void
  local_advance_concentration_rhs(
    const MatrixFree<dim, double> &              data,
    VectorType &                                 dst,
    const VectorType &                           src,
    const std::pair<unsigned int, unsigned int> &cell_range);

  /**
   * Parameters
   */
  const LevelSetOKZSolverAdvanceConcentrationParameter parameters; // [i]

  /**
   * Vector section
   */
  VectorType &      solution;         // [o] new ls solution
  const VectorType &solution_old;     // [i] old ls solution
  const VectorType &solution_old_old; // [i] old ls solution
  VectorType &      increment;        // [-] temp
  VectorType &      rhs;              // [-] temp

  const VectorType &vel_solution;         // [i] new velocity solution
  const VectorType &vel_solution_old;     // [i] old velocity solution
  const VectorType &vel_solution_old_old; // [i] old velocity solution

  /**
   * MatrixFree
   */
  const MatrixFree<dim> &          matrix_free; // [i]
  const AffineConstraints<double> &constraints; // [i]

  /**
   * Utility
   */
  const ConditionalOStream &pcout;         // [i]
  TimeStepping              time_stepping; // [i]

  /**
   * Physics section
   */
  const double &                                global_omega_diameter;          // [i]
  const AlignedVector<VectorizedArray<double>> &cell_diameters;                 // [i]
  const LevelSetOKZSolverAdvanceConcentrationBoundaryDescriptor<dim> boundary;  // [i]
  AlignedVector<VectorizedArray<double>> &artificial_viscosities;               // [-] ???
  double &                                global_max_velocity;                  // [o]
  AlignedVector<Tensor<1, dim, VectorizedArray<double>>> &evaluated_convection; // [o]

  /**
   * Solver section
   */
  const DiagonalPreconditioner<double> &preconditioner; // [i] preconditioner
};

#endif
