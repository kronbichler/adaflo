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
 * Parameters of the avection-concentration operator.
 */
struct LevelSetOKZSolverAdvanceConcentrationParameter
{
  /**
   * TODO: needed? this is equivalent to `fe.tenosor_degree()+1`?
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
};

template <int dim>
class LevelSetOKZSolverAdvanceConcentration
{
public:
  LevelSetOKZSolverAdvanceConcentration(
    LinearAlgebra::distributed::Vector<double> &      solution,
    const LinearAlgebra::distributed::Vector<double> &solution_old,
    const LinearAlgebra::distributed::Vector<double> &solution_old_old,
    LinearAlgebra::distributed::Vector<double> &      increment,
    LinearAlgebra::distributed::Vector<double> &      rhs,
    double &                                          global_omega_diameter,
    AlignedVector<VectorizedArray<double>> &          cell_diameters,

    const AffineConstraints<double> &                       constraints,
    const ConditionalOStream &                              pcout,
    const TimeStepping &                                    time_stepping,
    std::shared_ptr<helpers::BoundaryDescriptor<dim>> &     boundary,
    const MatrixFree<dim> &                                 matrix_free,
    const std::shared_ptr<TimerOutput> &                    timer,
    const NavierStokes<dim> &                               navier_stokes,
    const LevelSetOKZSolverAdvanceConcentrationParameter &  parameters,
    AlignedVector<VectorizedArray<double>> &                artificial_viscosities,
    double &                                                global_max_velocity,
    const DiagonalPreconditioner<double> &                  preconditioner,
    AlignedVector<Tensor<1, dim, VectorizedArray<double>>> &evaluated_convection)
    : solution(solution)
    , solution_old(solution_old)
    , solution_old_old(solution_old_old)
    , increment(increment)
    , rhs(rhs)
    , global_omega_diameter(global_omega_diameter)
    , cell_diameters(cell_diameters)
    , constraints(constraints)
    , pcout(pcout)
    , time_stepping(time_stepping)
    , boundary(boundary)
    , matrix_free(matrix_free)
    , timer(timer)
    , navier_stokes(navier_stokes)
    , parameters(parameters)
    , artificial_viscosities(artificial_viscosities)
    , global_max_velocity(global_max_velocity)
    , preconditioner(preconditioner)
    , evaluated_convection(evaluated_convection)
  {}

  // TODO: make utility function?
  double
  get_maximal_velocity(const DoFHandler<dim> &dof_handler) const
  {
    const QIterated<dim> quadrature_formula(QTrapez<1>(),
                                            dof_handler.get_fe().tensor_degree() + 1);

    FEValues<dim> fe_values(dof_handler.get_fe(), quadrature_formula, update_values);
    std::vector<Tensor<1, dim>> velocity_values(quadrature_formula.size());

    const FEValuesExtractors::Vector velocities(0);

    double max_velocity = 0;

    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          fe_values.reinit(cell);
          fe_values[velocities].get_function_values(navier_stokes.solution.block(0),
                                                    velocity_values);

          for (const auto q : fe_values.quadrature_point_indices())
            max_velocity = std::max(max_velocity, velocity_values[q].norm());
        }

    return Utilities::MPI::max(max_velocity, get_communicator(dof_handler));
  }

  virtual void
  advance_concentration();

  void
  advance_concentration_vmult(
    LinearAlgebra::distributed::Vector<double> &      dst,
    const LinearAlgebra::distributed::Vector<double> &src) const;

private:
  template <int ls_degree, int velocity_degree>
  void
  local_advance_concentration(
    const MatrixFree<dim, double> &                   data,
    LinearAlgebra::distributed::Vector<double> &      dst,
    const LinearAlgebra::distributed::Vector<double> &src,
    const std::pair<unsigned int, unsigned int> &     cell_range) const;

  template <int ls_degree, int velocity_degree>
  void
  local_advance_concentration_rhs(
    const MatrixFree<dim, double> &                   data,
    LinearAlgebra::distributed::Vector<double> &      dst,
    const LinearAlgebra::distributed::Vector<double> &src,
    const std::pair<unsigned int, unsigned int> &     cell_range);


  static const unsigned int dof_index_ls  = 2; // TODO: make input variables
  static const unsigned int dof_index_vel = 0; //
  static const unsigned int quad_index    = 2; //

  using VectorType = LinearAlgebra::distributed::Vector<double>;

  VectorType &      solution;         // [o] new ls solution
  const VectorType &solution_old;     // [i] old ls solution
  const VectorType &solution_old_old; // [i] old ls solution
  VectorType &      increment;        // [-] temp
  VectorType &      rhs;              // [-] temp

  double &                                global_omega_diameter;
  AlignedVector<VectorizedArray<double>> &cell_diameters;


  const AffineConstraints<double> &constraints;
  const ConditionalOStream &       pcout;
  const TimeStepping &             time_stepping;



  std::shared_ptr<helpers::BoundaryDescriptor<dim>> &boundary;


  const MatrixFree<dim> &matrix_free;

  const std::shared_ptr<TimerOutput> &timer;
  const NavierStokes<dim> &           navier_stokes;

  const LevelSetOKZSolverAdvanceConcentrationParameter parameters;

  AlignedVector<VectorizedArray<double>> &artificial_viscosities;
  double &                                global_max_velocity;

  const DiagonalPreconditioner<double> &                  preconditioner;
  AlignedVector<Tensor<1, dim, VectorizedArray<double>>> &evaluated_convection;
};

#endif
