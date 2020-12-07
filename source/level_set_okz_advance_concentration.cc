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

#include <deal.II/base/timer.h>

#include <deal.II/lac/solver_bicgstab.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_gmres.h>

#include <deal.II/matrix_free/fe_evaluation.h>

#include <deal.II/numerics/vector_tools.h>

#include <adaflo/level_set_okz_advance_concentration.h>
#include <adaflo/level_set_okz_template_instantations_adv.h>

namespace
{
  /**
   * Compute maximal velocity for a given vector and the corresponding
   * dof-handler object.
   */
  template <int dim, typename VectorType>
  double
  get_maximal_velocity(const DoFHandler<dim> &dof_handler, const VectorType solution)
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
          fe_values[velocities].get_function_values(solution, velocity_values);

          for (const auto q : fe_values.quadrature_point_indices())
            max_velocity = std::max(max_velocity, velocity_values[q].norm());
        }

    return Utilities::MPI::max(max_velocity, get_communicator(dof_handler));
  }
} // namespace



template <int dim>
LevelSetOKZSolverAdvanceConcentration<dim>::LevelSetOKZSolverAdvanceConcentration(
  VectorType &                                                   solution,
  const VectorType &                                             solution_old,
  const VectorType &                                             solution_old_old,
  VectorType &                                                   increment,
  VectorType &                                                   rhs,
  const VectorType &                                             vel_solution,
  const VectorType &                                             vel_solution_old,
  const VectorType &                                             vel_solution_old_old,
  const double &                                                 global_omega_diameter,
  const AlignedVector<VectorizedArray<double>> &                 cell_diameters,
  const AffineConstraints<double> &                              constraints,
  const ConditionalOStream &                                     pcout,
  const LevelSetOKZSolverAdvanceConcentrationBoundaryDescriptor &boundary,
  const MatrixFree<dim> &                                        matrix_free,
  const LevelSetOKZSolverAdvanceConcentrationParameter &         parameters,
  AlignedVector<VectorizedArray<double>> &                       artificial_viscosities,
  double &                                                       global_max_velocity,
  const DiagonalPreconditioner<double> &                         preconditioner,
  AlignedVector<Tensor<1, dim, VectorizedArray<double>>> &       evaluated_convection)
  : parameters(parameters)
  , solution(solution)
  , solution_old(solution_old)
  , solution_old_old(solution_old_old)
  , increment(increment)
  , rhs(rhs)
  , vel_solution(vel_solution)
  , vel_solution_old(vel_solution_old)
  , vel_solution_old_old(vel_solution_old_old)
  , matrix_free(matrix_free)
  , constraints(constraints)
  , pcout(pcout)
  , time_stepping(parameters.time)
  , global_omega_diameter(global_omega_diameter)
  , cell_diameters(cell_diameters)
  , boundary(boundary)
  , artificial_viscosities(artificial_viscosities)
  , global_max_velocity(global_max_velocity)
  , evaluated_convection(evaluated_convection)
  , preconditioner(preconditioner)
{}



template <int dim>
template <int ls_degree, int velocity_degree>
void
LevelSetOKZSolverAdvanceConcentration<dim>::local_advance_concentration(
  const MatrixFree<dim, double> &              data,
  VectorType &                                 dst,
  const VectorType &                           src,
  const std::pair<unsigned int, unsigned int> &cell_range) const
{
  // The second input argument below refers to which constrains should be used,
  // 2 means constraints (for LS-function)
  FEEvaluation<dim, ls_degree, 2 * ls_degree, 1> ls_values(data,
                                                           parameters.dof_index_ls,
                                                           parameters.quad_index);
  for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      const Tensor<1, dim, VectorizedArray<double>> *velocities =
        &evaluated_convection[cell * ls_values.n_q_points];
      ls_values.reinit(cell);

      ls_values.gather_evaluate(src, true, true);

      for (unsigned int q = 0; q < ls_values.n_q_points; ++q)
        {
          const VectorizedArray<double>                 ls_val = ls_values.get_value(q);
          const Tensor<1, dim, VectorizedArray<double>> ls_grad =
            ls_values.get_gradient(q);
          ls_values.submit_value(ls_val * this->time_stepping.weight() +
                                   ls_grad * velocities[q],
                                 q);
          if (this->parameters.convection_stabilization)
            ls_values.submit_gradient(artificial_viscosities[cell] * ls_grad, q);
        }
      ls_values.integrate_scatter(true, this->parameters.convection_stabilization, dst);
    }
}



template <int dim>
template <int ls_degree, int velocity_degree>
void
LevelSetOKZSolverAdvanceConcentration<dim>::local_advance_concentration_rhs(
  const MatrixFree<dim, double> &data,
  VectorType &                   dst,
  const VectorType &,
  const std::pair<unsigned int, unsigned int> &cell_range)
{
  // The second input argument below refers to which constrains should be used,
  // 2 means constraints (for LS-function) and 0 means
  // &navier_stokes.get_constraints_u()
  FEEvaluation<dim, ls_degree, 2 * ls_degree, 1> ls_values(data,
                                                           parameters.dof_index_ls,
                                                           parameters.quad_index);
  FEEvaluation<dim, ls_degree, 2 * ls_degree, 1> ls_values_old(data,
                                                               parameters.dof_index_ls,
                                                               parameters.quad_index);
  FEEvaluation<dim, ls_degree, 2 * ls_degree, 1> ls_values_old_old(
    data, parameters.dof_index_ls, parameters.quad_index);
  FEEvaluation<dim, velocity_degree, 2 * ls_degree, dim> vel_values(
    data, parameters.dof_index_vel, parameters.quad_index);
  FEEvaluation<dim, velocity_degree, 2 * ls_degree, dim> vel_values_old(
    data, parameters.dof_index_vel, parameters.quad_index);
  FEEvaluation<dim, velocity_degree, 2 * ls_degree, dim> vel_values_old_old(
    data, parameters.dof_index_vel, parameters.quad_index);

  typedef VectorizedArray<double> vector_t;

  for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      ls_values.reinit(cell);
      ls_values_old.reinit(cell);
      ls_values_old_old.reinit(cell);
      vel_values.reinit(cell);
      vel_values_old.reinit(cell);
      vel_values_old_old.reinit(cell);

      vel_values.read_dof_values_plain(vel_solution);
      vel_values_old.read_dof_values_plain(vel_solution_old);
      vel_values_old_old.read_dof_values_plain(vel_solution_old_old);
      ls_values.read_dof_values_plain(this->solution);
      ls_values_old.read_dof_values_plain(this->solution_old);
      ls_values_old_old.read_dof_values_plain(this->solution_old_old);

      vel_values.evaluate(true, false);
      vel_values_old.evaluate(true, false);
      vel_values_old_old.evaluate(true, false);
      ls_values.evaluate(true, true);
      ls_values_old.evaluate(true, true);
      ls_values_old_old.evaluate(true, true);

      if (this->parameters.convection_stabilization)
        {
          vector_t max_residual = vector_t(), max_velocity = vector_t();
          for (unsigned int q = 0; q < ls_values.n_q_points; ++q)
            {
              // compute residual of concentration equation
              Tensor<1, dim, vector_t> u =
                (vel_values_old.get_value(q) + vel_values_old_old.get_value(q));
              vector_t dc_dt =
                (ls_values_old.get_value(q) - ls_values_old_old.get_value(q)) /
                this->time_stepping.old_step_size();
              vector_t residual = std::abs(
                dc_dt +
                u * (ls_values_old.get_gradient(q) + ls_values_old_old.get_gradient(q)) *
                  0.25);
              max_residual = std::max(residual, max_residual);
              max_velocity = std::max(std::sqrt(u * u), max_velocity);
            }
          double global_scaling = global_max_velocity * 2 * this->global_omega_diameter;
          const vector_t cell_diameter = this->cell_diameters[cell];

          artificial_viscosities[cell] =
            0.03 * max_velocity * cell_diameter *
            std::min(make_vectorized_array(1.), 1. * max_residual / global_scaling);
        }

      Tensor<1, dim, VectorizedArray<double>> *velocities =
        &evaluated_convection[cell * ls_values.n_q_points];

      for (unsigned int q = 0; q < ls_values.n_q_points; ++q)
        {
          // compute right hand side
          vector_t old_value =
            this->time_stepping.weight_old() * ls_values_old.get_value(q);
          if (this->time_stepping.scheme() == TimeSteppingParameters::Scheme::bdf_2 &&
              this->time_stepping.step_no() > 1)
            old_value +=
              this->time_stepping.weight_old_old() * ls_values_old_old.get_value(q);
          const vector_t                 ls_val  = ls_values.get_value(q);
          const Tensor<1, dim, vector_t> ls_grad = ls_values.get_gradient(q);
          vector_t residual = -(ls_val * this->time_stepping.weight() +
                                vel_values.get_value(q) * ls_grad + old_value);
          ls_values.submit_value(residual, q);
          if (this->parameters.convection_stabilization)
            ls_values.submit_gradient(-artificial_viscosities[cell] * ls_grad, q);
          velocities[q] = vel_values.get_value(q);
        }
      ls_values.integrate_scatter(true, this->parameters.convection_stabilization, dst);
    }
}



template <int dim>
void
LevelSetOKZSolverAdvanceConcentration<dim>::advance_concentration_vmult(
  VectorType &      dst,
  const VectorType &src) const
{
  dst = 0.;
#define OPERATION(c_degree, u_degree)                                 \
  this->matrix_free.cell_loop(                                        \
    &LevelSetOKZSolverAdvanceConcentration<                           \
      dim>::template local_advance_concentration<c_degree, u_degree>, \
    this,                                                             \
    dst,                                                              \
    src)

  EXPAND_OPERATIONS(OPERATION);
#undef OPERATION


  if (this->parameters.convection_stabilization)
    {
      const auto &dof_handler =
        this->matrix_free.get_dof_handler(parameters.dof_index_ls);
      const auto &fe = dof_handler.get_fe();

      // Boundary part of stabilization-term:
      FEFaceValues<dim> fe_face_values(
        fe,
        this->matrix_free.get_face_quadrature(parameters.quad_index),
        update_values | update_gradients | update_JxW_values | update_normal_vectors);
      Vector<double>                       cell_rhs(fe.dofs_per_cell);
      std::vector<types::global_dof_index> local_dof_indices(fe.dofs_per_cell);
      std::vector<Tensor<1, dim>> local_gradients(fe_face_values.get_quadrature().size());
      src.update_ghost_values();

      for (unsigned int mcell = 0; mcell < this->matrix_free.n_cell_batches(); ++mcell)
        for (unsigned int v = 0;
             v < this->matrix_free.n_active_entries_per_cell_batch(mcell);
             ++v)
          {
            typename DoFHandler<dim>::active_cell_iterator cell =
              this->matrix_free.get_cell_iterator(mcell, v, 2);
            cell_rhs = 0;

            for (const auto &face : cell->face_iterators())
              {
                if (face->at_boundary() == false)
                  continue;

                if (this->boundary.symmetry.find(face->boundary_id()) !=
                    this->boundary.symmetry.end())
                  continue;

                fe_face_values.reinit(cell, face);
                fe_face_values.get_function_gradients(src, local_gradients);
                for (const auto i : fe_face_values.dof_indices())
                  for (const auto q : fe_face_values.quadrature_point_indices())
                    cell_rhs(i) +=
                      -((fe_face_values.shape_value(i, q) *
                         fe_face_values.normal_vector(q) *
                         artificial_viscosities[mcell][v] * local_gradients[q]) *
                        fe_face_values.JxW(q));
              }

            cell->get_dof_indices(local_dof_indices);
            this->constraints.distribute_local_to_global(cell_rhs,
                                                         local_dof_indices,
                                                         dst);
          }

      dst.compress(VectorOperation::add);
    }

  for (unsigned int i = 0; i < this->matrix_free.get_constrained_dofs(2).size(); ++i)
    dst.local_element(this->matrix_free.get_constrained_dofs(2)[i]) =
      preconditioner.get_vector().local_element(
        this->matrix_free.get_constrained_dofs(2)[i]) *
      src.local_element(this->matrix_free.get_constrained_dofs(2)[i]);
}



template <int dim, typename VectorType>
struct AdvanceConcentrationMatrix
{
  AdvanceConcentrationMatrix(const LevelSetOKZSolverAdvanceConcentration<dim> &problem)
    : problem(problem)
  {}

  void
  vmult(VectorType &dst, const VectorType &src) const
  {
    problem.advance_concentration_vmult(dst, src);
  }

  const LevelSetOKZSolverAdvanceConcentration<dim> &problem;
};



// @sect4{LevelSetOKZSolverAdvanceConcentration::advance_concentration}
template <int dim>
void
LevelSetOKZSolverAdvanceConcentration<dim>::advance_concentration()
{
  const auto &mapping     = *this->matrix_free.get_mapping_info().mapping;
  const auto &dof_handler = this->matrix_free.get_dof_handler(parameters.dof_index_ls);
  const auto &fe          = dof_handler.get_fe();

  // apply boundary values
  {
    std::map<types::boundary_id, const Function<dim> *> dirichlet;

    Functions::ConstantFunction<dim> plus_func(1., 1);
    for (const auto bid : this->boundary.fluid_type_plus)
      dirichlet[bid] = &plus_func;

    Functions::ConstantFunction<dim> minus_func(-1., 1);
    for (const auto bid : this->boundary.fluid_type_minus)
      dirichlet[bid] = &minus_func;

    std::map<types::global_dof_index, double> boundary_values;
    VectorTools::interpolate_boundary_values(mapping,
                                             dof_handler,
                                             dirichlet,
                                             boundary_values);

    for (const auto &it : boundary_values)
      if (this->solution.locally_owned_elements().is_element(it.first))
        this->solution(it.first) = it.second;
    this->solution.update_ghost_values();
  }

  // compute right hand side
  global_max_velocity =
    get_maximal_velocity(matrix_free.get_dof_handler(parameters.dof_index_vel),
                         vel_solution);
  rhs = 0;

#define OPERATION(c_degree, u_degree)                                     \
  this->matrix_free.cell_loop(                                            \
    &LevelSetOKZSolverAdvanceConcentration<                               \
      dim>::template local_advance_concentration_rhs<c_degree, u_degree>, \
    this,                                                                 \
    rhs,                                                                  \
    this->solution)

  EXPAND_OPERATIONS(OPERATION);
#undef OPERATION

  AdvanceConcentrationMatrix<dim, VectorType> matrix(*this);



  if (this->parameters.convection_stabilization)
    {
      // Boundary part of stabilization-term:
      FEFaceValues<dim> fe_face_values(
        mapping,
        fe,
        this->matrix_free.get_face_quadrature(parameters.quad_index),
        update_values | update_gradients | update_JxW_values | update_normal_vectors);

      Vector<double>                       cell_rhs(fe.dofs_per_cell);
      std::vector<types::global_dof_index> local_dof_indices(fe.dofs_per_cell);
      std::vector<Tensor<1, dim>> local_gradients(fe_face_values.get_quadrature().size());

      for (unsigned int mcell = 0; mcell < this->matrix_free.n_cell_batches(); ++mcell)
        for (unsigned int v = 0;
             v < this->matrix_free.n_active_entries_per_cell_batch(mcell);
             ++v)
          {
            typename DoFHandler<dim>::active_cell_iterator cell =
              this->matrix_free.get_cell_iterator(mcell, v, 2);
            cell_rhs = 0;

            for (const auto face : cell->face_iterators())
              {
                if (face->at_boundary() == false)
                  continue;

                if (this->boundary.symmetry.find(face->boundary_id()) !=
                    this->boundary.symmetry.end())
                  continue;

                fe_face_values.reinit(cell, face);
                fe_face_values.get_function_gradients(this->solution, local_gradients);

                for (const auto i : fe_face_values.dof_indices())
                  for (const auto q : fe_face_values.quadrature_point_indices())
                    cell_rhs(i) +=
                      ((fe_face_values.shape_value(i, q) *
                        fe_face_values.normal_vector(q) *
                        artificial_viscosities[mcell][v] * local_gradients[q]) *
                       fe_face_values.JxW(q));
              }

            cell->get_dof_indices(local_dof_indices);
            this->constraints.distribute_local_to_global(cell_rhs,
                                                         local_dof_indices,
                                                         this->rhs);
          }
      this->rhs.compress(VectorOperation::add);
    }


  // solve linear system with Bicgstab (non-symmetric system!)
  unsigned int n_iterations     = 0;
  double       initial_residual = 0.;
  try
    {
      ReductionControl control(30, 0.05 * this->parameters.tol_nl_iteration, 1e-8);
      SolverBicgstab<VectorType>::AdditionalData bicg_data;
      bicg_data.exact_residual = false;
      SolverBicgstab<VectorType> solver(control, bicg_data);
      increment = 0;
      solver.solve(matrix, increment, rhs, preconditioner);
      n_iterations     = control.last_step();
      initial_residual = control.initial_value();
    }
  catch (const SolverControl::NoConvergence &)
    {
      // GMRES is typically slower but much more robust
      ReductionControl control(3000, 0.05 * this->parameters.tol_nl_iteration, 1e-8);
      SolverGMRES<VectorType> solver(control);
      solver.solve(matrix, increment, rhs, preconditioner);
      n_iterations = 30 + control.last_step();
    }
  if (!this->parameters.do_iteration)
    this->pcout << "  Concentration advance: advect [" << initial_residual << "/"
                << n_iterations << "]";

  this->constraints.distribute(increment);
  this->solution += increment;
  this->solution.update_ghost_values();
}


template class LevelSetOKZSolverAdvanceConcentration<2>;
template class LevelSetOKZSolverAdvanceConcentration<3>;
