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

template <int dim>
class LevelSetOKZSolverAdvanceConcentration
{
public:
  LevelSetOKZSolverAdvanceConcentration(
    const LinearAlgebra::distributed::Vector<double> &solution_old,
    const LinearAlgebra::distributed::Vector<double> &solution_old_old,
    LinearAlgebra::distributed::Vector<double> &      solution_update,
    LinearAlgebra::distributed::Vector<double> &      solution,
    LinearAlgebra::distributed::Vector<double> &      system_rhs,
    Triangulation<dim> &                              triangulation,
    double &                                          global_omega_diameter,
    AlignedVector<VectorizedArray<double>> &          cell_diameters,

    const AffineConstraints<double> &                       constraints,
    const ConditionalOStream &                              pcout,
    const TimeStepping &                                    time_stepping,
    std::shared_ptr<helpers::BoundaryDescriptor<dim>> &     boundary,
    Mapping<dim> &                                          mapping,
    DoFHandler<dim> &                                       dof_handler,
    const std::shared_ptr<FiniteElement<dim>> &             fe,
    const MatrixFree<dim> &                                 matrix_free,
    const std::shared_ptr<TimerOutput> &                    timer,
    const NavierStokes<dim> &                               navier_stokes,
    const FlowParameters &                                  parameters,
    AlignedVector<VectorizedArray<double>> &                artificial_viscosities,
    double &                                                global_max_velocity,
    const DiagonalPreconditioner<double> &                  preconditioner,
    AlignedVector<Tensor<1, dim, VectorizedArray<double>>> &evaluated_convection)
    : solution_old(solution_old)
    , solution_old_old(solution_old_old)
    , solution_update(solution_update)
    , solution(solution)
    , system_rhs(system_rhs)
    , triangulation(triangulation)
    , global_omega_diameter(global_omega_diameter)
    , cell_diameters(cell_diameters)
    , constraints(constraints)
    , pcout(pcout)
    , time_stepping(time_stepping)
    , boundary(boundary)
    , mapping(mapping)
    , dof_handler(dof_handler)
    , fe(fe)
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
  get_maximal_velocity() const
  {
    const QIterated<dim> quadrature_formula(QTrapez<1>(), parameters.velocity_degree + 1);
    const unsigned int   n_q_points = quadrature_formula.size();

    FEValues<dim> fe_values(navier_stokes.get_fe_u(), quadrature_formula, update_values);
    std::vector<Tensor<1, dim>> velocity_values(n_q_points);

    const FEValuesExtractors::Vector velocities(0);

    double max_velocity = 0;

    typename DoFHandler<dim>::active_cell_iterator
      cell = navier_stokes.get_dof_handler_u().begin_active(),
      endc = navier_stokes.get_dof_handler_u().end();
    for (; cell != endc; ++cell)
      if (cell->is_locally_owned())
        {
          fe_values.reinit(cell);
          fe_values[velocities].get_function_values(navier_stokes.solution.block(0),
                                                    velocity_values);

          for (unsigned int q = 0; q < n_q_points; ++q)
            max_velocity = std::max(max_velocity, velocity_values[q].norm());
        }

    return Utilities::MPI::max(max_velocity, get_communicator(triangulation));
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


  const LinearAlgebra::distributed::Vector<double> &solution_old;
  const LinearAlgebra::distributed::Vector<double> &solution_old_old;
  LinearAlgebra::distributed::Vector<double> &      solution_update;
  LinearAlgebra::distributed::Vector<double> &      solution;
  LinearAlgebra::distributed::Vector<double> &      system_rhs;

  Triangulation<dim> &                    triangulation;
  double &                                global_omega_diameter;
  AlignedVector<VectorizedArray<double>> &cell_diameters;


  const AffineConstraints<double> &constraints;
  const ConditionalOStream &       pcout;
  const TimeStepping &             time_stepping;



  std::shared_ptr<helpers::BoundaryDescriptor<dim>> &boundary;
  Mapping<dim> &                                     mapping;
  DoFHandler<dim> &                                  dof_handler;
  const std::shared_ptr<FiniteElement<dim>> &        fe;


  const MatrixFree<dim> &matrix_free;

  const std::shared_ptr<TimerOutput> &timer;
  const NavierStokes<dim> &           navier_stokes;
  const FlowParameters &              parameters;


  AlignedVector<VectorizedArray<double>> &artificial_viscosities;
  double &                                global_max_velocity;

  const DiagonalPreconditioner<double> &                  preconditioner;
  AlignedVector<Tensor<1, dim, VectorizedArray<double>>> &evaluated_convection;
};

#endif
