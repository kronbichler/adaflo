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

#include <deal.II/base/index_set.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_q_iso_q1.h>

#include <deal.II/grid/tria.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/solver_bicgstab.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <deal.II/numerics/vector_tools.h>

#include <adaflo/level_set_okz_matrix.h>
#include <adaflo/util.h>

#include <fstream>
#include <iostream>


using namespace dealii;


template <int dim>
adaflo::LevelSetOKZMatrixSolver<dim>::LevelSetOKZMatrixSolver(
  const FlowParameters &parameters_in,
  Triangulation<dim>   &tria_in)
  : LevelSetBaseAlgorithm<dim>(parameters_in, tria_in)
{}



template <int dim>
void
adaflo::LevelSetOKZMatrixSolver<dim>::transform_distance_function(
  LinearAlgebra::distributed::Vector<double> &vector) const
{
  Assert(this->epsilon_used > 0, ExcInternalError());
  for (unsigned int i = 0; i < vector.locally_owned_size(); i++)
    vector.local_element(i) =
      -std::tanh(vector.local_element(i) / (2. * this->epsilon_used));
}



// @sect4{LevelSetOKZMatrixSolver::make_grid_and_dofs}
template <int dim>
void
adaflo::LevelSetOKZMatrixSolver<dim>::initialize_data_structures()
{
  system_matrix.clear();
  this->LevelSetBaseAlgorithm<dim>::initialize_data_structures();

  normal_calculated = false;

  IndexSet locally_relevant_dofs(this->dof_handler.n_dofs());
  DoFTools::extract_locally_relevant_dofs(this->dof_handler, locally_relevant_dofs);
  DynamicSparsityPattern dsp(this->dof_handler.n_dofs(),
                             this->dof_handler.n_dofs(),
                             locally_relevant_dofs);
  DoFTools::make_sparsity_pattern(this->dof_handler, dsp, this->constraints, true);
  system_matrix.reinit(this->dof_handler.locally_owned_dofs(),
                       dsp,
                       this->triangulation.get_communicator(),
                       true);
}



template <int dim>
void
adaflo::LevelSetOKZMatrixSolver<dim>::compute_force()
{
  compute_heaviside();
  compute_curvature();

  TimerOutput::Scope timer(*this->timer, "Compute force.");

  this->compute_density_on_faces();

  this->navier_stokes.user_rhs = 0;

  FEValues<dim> fe_values(this->mapping,
                          this->navier_stokes.get_fe_u(),
                          this->matrix_free.get_quadrature(0),
                          update_values | update_JxW_values);

  const unsigned int dofs_per_cell = this->navier_stokes.get_fe_u().dofs_per_cell;
  const unsigned int dofs_per_p    = this->navier_stokes.get_fe_p().dofs_per_cell;
  const unsigned int n_q_points    = fe_values.get_quadrature().size();
  Vector<double>     local_rhs(dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
  std::vector<types::global_dof_index> ls_dofs(this->fe->dofs_per_cell);

  const FEValuesExtractors::Vector velocities(0);

  // Now we define structures for evaluating the level set solutions <i>c</i>
  // and <i>w</i> as they were functions. Project the gradient term $\nabla C$
  // onto the pressure space in order to balance the surface tension force
  // with pressure

  FullMatrix<double>       interpolate_c_onto_p(dofs_per_p, this->fe->dofs_per_cell);
  Table<2, Tensor<1, dim>> gradients_c(this->fe->dofs_per_cell, n_q_points);
  {
    FEValues<dim> interpolate_for_p(
      *this->fe, this->navier_stokes.get_fe_p().get_unit_support_points(), update_values);
    interpolate_for_p.reinit(this->triangulation.begin_active());
    for (unsigned int j = 0; j < this->fe->dofs_per_cell; ++j)
      for (unsigned int i = 0; i < dofs_per_p; ++i)
        interpolate_c_onto_p(i, j) = interpolate_for_p.shape_value(j, i);
    FEValues<dim> interpolate_p(this->navier_stokes.get_fe_p(),
                                this->matrix_free.get_quadrature(0),
                                update_gradients | update_jacobians);
    interpolate_p.reinit(this->triangulation.begin_active());
    for (unsigned int i = 0; i < dofs_per_p; ++i)
      for (unsigned int q = 0; q < n_q_points; ++q)
        {
          Tensor<2, dim> jac = interpolate_p.jacobian(q);
          gradients_c(i, q)  = transpose(jac) * interpolate_p.shape_grad(i, q);
        }
  }

  FEValues<dim> ls_values(this->mapping,
                          *this->fe,
                          this->matrix_free.get_quadrature(0),
                          update_values | update_gradients | update_inverse_jacobians);

  std::vector<double>         c_values(n_q_points);
  std::vector<double>         c_values_old(n_q_points);
  std::vector<Tensor<1, dim>> c_gradients(n_q_points);
  std::vector<Tensor<1, dim>> c_gradients_old(n_q_points);
  std::vector<double>         w_values(n_q_points);
  std::vector<double>         w_values_old(n_q_points);
  std::vector<Tensor<1, dim>> w_gradients(n_q_points);
  Vector<double>              c_val(this->fe->dofs_per_cell);

  const double tau1 = this->time_stepping.step_no() > 1 ? this->time_stepping.tau1() : 1;
  const double tau2 = this->time_stepping.step_no() > 1 ? this->time_stepping.tau2() : 0;

  this->navier_stokes.user_rhs = 0;
  const bool use_variable_parameters =
    this->parameters.density_diff != 0 || this->parameters.viscosity_diff != 0;

  // loop over all cells and treat only those that are close to the interface
  // and give a contribution to the force
  for (unsigned int mcell = 0; mcell < this->matrix_free.n_cell_batches(); ++mcell)
    for (unsigned int vec = 0;
         vec < this->matrix_free.n_active_entries_per_cell_batch(mcell);
         ++vec)
      {
        local_rhs = 0;

        VectorizedArray<double> *densities =
          use_variable_parameters ?
            this->navier_stokes.get_matrix().begin_densities(mcell) :
            0;
        VectorizedArray<double> *viscosities =
          use_variable_parameters ?
            this->navier_stokes.get_matrix().begin_viscosities(mcell) :
            0;
        typename DoFHandler<dim>::active_cell_iterator
          cell = this->matrix_free.get_cell_iterator(mcell, vec, 0),
          ls_cell(&this->triangulation, cell->level(), cell->index(), &this->dof_handler);

        // initialize FEValues objects
        fe_values.reinit(cell);
        ls_values.reinit(ls_cell);

        // find gradients of the level set function (corresponding to
        // approximation of $\delta_\Gamma \ve n$). Possibly, interpolate this
        // onto the pressure space
        ls_values.get_function_values(this->parameters.surface_tension_from_heaviside ?
                                        this->heaviside :
                                        this->solution.block(0),
                                      c_values);
        if (this->parameters.interpolate_grad_onto_pressure == false)
          ls_values.get_function_gradients(
            this->parameters.surface_tension_from_heaviside ? this->heaviside :
                                                              this->solution.block(0),
            c_gradients);
        else
          {
            ls_cell->get_interpolated_dof_values(
              this->parameters.surface_tension_from_heaviside ? this->heaviside :
                                                                this->solution.block(0),
              c_val);
            std::fill(c_gradients.begin(), c_gradients.end(), Tensor<1, dim>());
            for (unsigned int j = 0; j < dofs_per_p; ++j)
              {
                double c_val_on_p = 0;
                for (unsigned int i = 0; i < this->fe->dofs_per_cell; i++)
                  c_val_on_p += c_val(i) * interpolate_c_onto_p(j, i);
                for (unsigned int q = 0; q < n_q_points; ++q)
                  c_gradients[q] += c_val_on_p * gradients_c(j, q);
              }
          }

        // extract curvature for the current cell's quadrature points
        ls_values.get_function_values(this->solution.block(1), w_values);

        // TODO: fix this with heaviside function (i.e., when using
        // Crank-Nicholson)

        // if the same information is needed from the old time step
        // (e.g. one-step theta scheme)
        if (this->time_stepping.tau2() > 1e-14 && this->time_stepping.step_no() > 1)
          {
            if (this->parameters.interpolate_grad_onto_pressure == false)
              ls_values.get_function_gradients(this->solution_old.block(0),
                                               c_gradients_old);
            else
              {
                ls_cell->get_interpolated_dof_values(this->solution_old.block(0), c_val);
                double c_val_on_p = 0;
                for (unsigned int i = 0; i < this->fe->dofs_per_cell; i++)
                  c_val_on_p += c_val(i) * interpolate_c_onto_p(0, i);
                for (unsigned int q = 0; q < n_q_points; ++q)
                  c_gradients_old[q] = c_val_on_p * gradients_c(0, q);
                for (unsigned int j = 1; j < dofs_per_p; ++j)
                  {
                    c_val_on_p = 0;
                    for (unsigned int i = 0; i < this->fe->dofs_per_cell; i++)
                      c_val_on_p += c_val(i) * interpolate_c_onto_p(j, i);
                    for (unsigned int q = 0; q < n_q_points; ++q)
                      c_gradients_old[q] += c_val_on_p * gradients_c(j, q);
                  }
                ls_values.get_function_values(this->solution_old.block(1), w_values_old);
                ls_values.get_function_values(this->solution_old.block(0), c_values_old);
              }
          }

        // loop over all quadrature points
        for (unsigned int q = 0; q < n_q_points; ++q)
          {
            double actual_rho = this->parameters.density;

            // compute variable density and viscosity
            if (use_variable_parameters)
              {
                if (this->parameters.surface_tension_from_heaviside)
                  {
                    actual_rho += this->parameters.density_diff * (c_values[q]);
                    densities[q][vec]   = actual_rho;
                    viscosities[q][vec] = this->parameters.viscosity +
                                          this->parameters.viscosity_diff * (c_values[q]);
                  }
                else
                  {
                    actual_rho += this->parameters.density_diff * 0.5 * (c_values[q] + 1);
                    densities[q][vec] = actual_rho;
                    viscosities[q][vec] =
                      this->parameters.viscosity +
                      this->parameters.viscosity_diff * 0.5 * (c_values[q] + 1);
                  }
              }

            // evaluate surface tension force
            const double   curvature = (tau1 * w_values[q] + tau2 * w_values_old[q]);
            Tensor<1, dim> grad_c =
              (this->parameters.surface_tension_from_heaviside ? 1. : 0.5) *
              (tau1 * c_gradients[q] + tau2 * c_gradients_old[q]);

            Tensor<1, dim> force =
              (grad_c * curvature * this->parameters.surface_tension);
            if (this->parameters.interpolate_grad_onto_pressure)
              force = transpose(Tensor<2, dim>(ls_values.inverse_jacobian(q))) * force;

            // add gravity in last component
            force[dim - 1] -= this->parameters.gravity * actual_rho;

            // test with all basis function on the cell
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                local_rhs(i) +=
                  (force * fe_values[velocities].value(i, q)) * fe_values.JxW(q);
              }
          }

        // write cell vector into the Navier--Stokes
        // vector
        cell->get_dof_indices(local_dof_indices);
        this->navier_stokes.get_constraints_u().distribute_local_to_global(
          local_rhs, local_dof_indices, this->navier_stokes.user_rhs.block(0));
      }
  this->navier_stokes.user_rhs.block(0).compress(VectorOperation::add);
}



// computation of a entropy-based viscosity for stabilizing the advection
// scheme (based on ideas by Guermond and Popov, see also the deal.II step-31
// tutorial program)
template <int dim>
double
compute_viscosity(const std::vector<double>         &old_temperature,
                  const std::vector<double>         &old_old_temperature,
                  const std::vector<Tensor<1, dim>> &old_temperature_grads,
                  const std::vector<Tensor<1, dim>> &old_old_temperature_grads,
                  const std::vector<Tensor<1, dim>> &old_velocity_values,
                  const std::vector<Tensor<1, dim>> &old_old_velocity_values,
                  const double                       time_step,
                  const double                       global_u_infty,
                  const double                       global_T_variation,
                  const double                       global_Omega_diameter,
                  const double                       cell_diameter)
{
  const double beta  = 0.02 * dim;
  const double alpha = 1;

  if (global_u_infty < 1e-5 || time_step < 1e-10)
    return 1e-2 * cell_diameter * std::fabs(global_u_infty);

  const unsigned int n_q_points = old_temperature.size();

  double max_residual = 0;
  double max_velocity = 0;

  for (unsigned int q = 0; q < n_q_points; ++q)
    {
      const Tensor<1, dim> u = (old_velocity_values[q] + old_old_velocity_values[q]) / 2;

      const double dT_dt = (old_temperature[q] - old_old_temperature[q]) / time_step;
      const double u_grad_T =
        u * (old_temperature_grads[q] + old_old_temperature_grads[q]) / 2;

      const double residual =
        std::abs((dT_dt + u_grad_T) *
                 std::pow((old_temperature[q] + old_old_temperature[q]) / 2, alpha - 1.));

      max_residual = std::max(residual, max_residual);
      max_velocity = std::max(std::sqrt(u * u), max_velocity);
    }

  const double c_R            = std::pow(2., (4. - 2 * alpha) / dim);
  const double global_scaling = c_R * global_u_infty * global_T_variation *
                                std::pow(global_Omega_diameter, alpha - 2.);

  return (beta * max_velocity *
          std::min(cell_diameter,
                   std::pow(cell_diameter, alpha) * max_residual / global_scaling));
}



// @sect4{LevelSetOKZMatrixSolver::advance_concentration}
template <int dim>
void
adaflo::LevelSetOKZMatrixSolver<dim>::advance_concentration()
{
  TimerOutput::Scope timer(*this->timer, "Advance concentration.");

  // apply boundary values
  {
    std::map<types::boundary_id, const Function<dim> *> dirichlet;

    for (const auto &b : this->boundary->fluid_type)
      dirichlet[b.first] = b.second.get();

    std::map<types::global_dof_index, double> boundary_values;
    VectorTools::interpolate_boundary_values(this->dof_handler,
                                             dirichlet,
                                             boundary_values);

    for (typename std::map<types::global_dof_index, double>::const_iterator it =
           boundary_values.begin();
         it != boundary_values.end();
         ++it)
      if (this->solution.block(0).in_local_range(it->first))
        this->solution.block(0)(it->first) = it->second;
    this->solution.block(0).update_ghost_values();
  }

  global_max_velocity = this->get_maximal_velocity();

  system_matrix                                         = 0;
  LinearAlgebra::distributed::Vector<double> &rhs       = this->system_rhs.block(0);
  LinearAlgebra::distributed::Vector<double> &increment = this->solution_update.block(0);
  rhs                                                   = 0;

  const double step_size_old = this->time_stepping.old_step_size();

  FEValues<dim> fe_values(this->mapping,
                          *this->fe,
                          this->matrix_free.get_quadrature(2),
                          update_values | update_gradients | update_inverse_jacobians |
                            update_JxW_values);

  FEValues<dim> ns_values(this->mapping,
                          this->navier_stokes.get_fe_u(),
                          fe_values.get_quadrature(),
                          update_values);

  const unsigned int dofs_per_cell = this->fe->dofs_per_cell;
  const unsigned int n_q_points    = fe_values.get_quadrature().size();

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  std::vector<Tensor<1, dim>> velocity_values(n_q_points);
  std::vector<Tensor<1, dim>> old_velocity_values(n_q_points);
  std::vector<Tensor<1, dim>> old_old_velocity_values(n_q_points);

  std::vector<double>         old_values(n_q_points);
  std::vector<double>         old_old_values(n_q_points);
  std::vector<Tensor<1, dim>> old_old_gradients(n_q_points);
  std::vector<Tensor<1, dim>> old_gradients(n_q_points);
  std::vector<double>         cur_values(n_q_points);
  std::vector<Tensor<1, dim>> cur_gradients(n_q_points);

  const double global_conc_range = 2.;

  const FEValuesExtractors::Vector velocities(0);

  typename DoFHandler<dim>::active_cell_iterator cell = this->dof_handler.begin_active(),
                                                 endc = this->dof_handler.end(),
                                                 ns_cell =
                                                   this->navier_stokes.get_dof_handler_u()
                                                     .begin_active();
  for (; cell != endc; ++cell, ++ns_cell)
    if (cell->is_locally_owned())
      {
        cell_matrix = 0;
        cell_rhs    = 0;

        fe_values.reinit(cell);
        ns_values.reinit(ns_cell);

        ns_values[velocities].get_function_values(this->navier_stokes.solution.block(0),
                                                  velocity_values);
        ns_values[velocities].get_function_values(
          this->navier_stokes.solution_old.block(0), old_velocity_values);
        fe_values.get_function_values(this->solution.block(0), cur_values);
        fe_values.get_function_gradients(this->solution.block(0), cur_gradients);
        fe_values.get_function_values(this->solution_old.block(0), old_values);
        fe_values.get_function_gradients(this->solution_old.block(0), old_gradients);
        if (this->time_stepping.step_no() > 1)
          {
            fe_values.get_function_values(this->solution_old_old.block(0),
                                          old_old_values);
            ns_values[velocities].get_function_values(
              this->navier_stokes.solution_old_old.block(0), old_old_velocity_values);
            fe_values.get_function_gradients(this->solution_old_old.block(0),
                                             old_old_gradients);
          }

        // if we use residual stabilization, there
        // will be no additional viscosity. otherwise,
        // compute it by the function given above

        const double viscosity = this->parameters.convection_stabilization ?
                                   compute_viscosity(old_values,
                                                     old_old_values,
                                                     old_gradients,
                                                     old_old_gradients,
                                                     old_velocity_values,
                                                     old_old_velocity_values,
                                                     step_size_old,
                                                     global_max_velocity,
                                                     global_conc_range,
                                                     this->global_omega_diameter,
                                                     cell->diameter()) :
                                   0.;


        for (unsigned int q = 0; q < n_q_points; ++q)
          {
            const Tensor<1, dim> &velocity = velocity_values[q];
            double old_value = -this->time_stepping.weight_old() * old_values[q];
            if (this->time_stepping.scheme() == TimeSteppingParameters::Scheme::bdf_2 &&
                this->time_stepping.step_no() > 1)
              old_value -= this->time_stepping.weight_old_old() * old_old_values[q];

            // computation of matrix entries based on the residual
            // stabilization
            if (this->parameters.convection_stabilization)
              {
                // The first thing to do is to calculate the stabilization
                // parameters. In this respect, we follow the suggestion in
                // Bazilevs et al. (Comp. Methods Appl. Mech. Engrg, 2007)
                double cell_tau_stab;
                {
                  const Tensor<2, dim> jac      = fe_values.inverse_jacobian(q);
                  const Tensor<2, dim> g_matrix = transpose(jac) * jac;

                  const double uGu = velocity * g_matrix * velocity;

                  cell_tau_stab = 1. / sqrt(4 * this->time_stepping.weight() *
                                              this->time_stepping.weight() +
                                            uGu);
                }

                // computation of matrix and right hand side
                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                  {
                    const double u_nabla_eta = velocity * fe_values.shape_grad(i, q);

                    for (unsigned int j = 0; j < dofs_per_cell; ++j)
                      {
                        const double u_res_eta =
                          velocity * fe_values.shape_grad(j, q) +
                          fe_values.shape_value(j, q) * this->time_stepping.weight();

                        cell_matrix(i, j) +=
                          (fe_values.shape_value(i, q) * this->time_stepping.weight() *
                             fe_values.shape_value(j, q) -
                           u_nabla_eta * fe_values.shape_value(j, q) +
                           u_nabla_eta * cell_tau_stab * u_res_eta) *
                          fe_values.JxW(q);
                      }

                    // TODO: need to put residual on rhs
                    cell_rhs(i) +=
                      ((fe_values.shape_value(i, q) + u_nabla_eta * cell_tau_stab) *
                       old_value) *
                      fe_values.JxW(q);
                  }
              }
            // stabilization with artificial viscosity (entropy-based)
            else
              {
                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                  {
                    const double u_nabla_eta = velocity * fe_values.shape_grad(i, q);

                    for (unsigned int j = 0; j < dofs_per_cell; ++j)
                      {
                        cell_matrix(i, j) +=
                          (fe_values.shape_value(i, q) * this->time_stepping.weight() *
                             fe_values.shape_value(j, q) -
                           u_nabla_eta * fe_values.shape_value(j, q) +
                           viscosity * fe_values.shape_grad(i, q) *
                             fe_values.shape_grad(j, q)) *
                          fe_values.JxW(q);
                      }

                    // put residual on RHS
                    cell_rhs(i) +=
                      (fe_values.shape_value(i, q) *
                         (old_value - this->time_stepping.weight() * cur_values[q]) +
                       u_nabla_eta * cur_values[q]
                       /*-
                         viscosity * fe_values.shape_grad(i,q) *
                         cur_gradients[q]*/
                       ) *
                      fe_values.JxW(q);
                  };
              }
          }

        cell->get_dof_indices(local_dof_indices);
        this->constraints.distribute_local_to_global(cell_matrix,
                                                     local_dof_indices,
                                                     system_matrix);
        this->constraints.distribute_local_to_global(cell_rhs, local_dof_indices, rhs);
      };

  rhs.compress(VectorOperation::add);
  system_matrix.compress(VectorOperation::add);

  TrilinosWrappers::PreconditionSSOR preconditioner;
  preconditioner.initialize(system_matrix);
  // solve linear system with Bicgstab (non-symmetric system!)
  SolverControl solver_control(6000,
                               std::max(1e-11 * rhs.l2_norm(),
                                        0.02 * this->parameters.tol_nl_iteration));
  SolverBicgstab<LinearAlgebra::distributed::Vector<double>>::AdditionalData bicg_data;
  bicg_data.exact_residual = false;
  SolverBicgstab<LinearAlgebra::distributed::Vector<double>> solver(solver_control,
                                                                    bicg_data);
  increment = 0;
  solver.solve(system_matrix, increment, rhs, preconditioner);
  if (!this->parameters.do_iteration)
    this->pcout << "  Concentration advance: advect (" << solver_control.last_step()
                << ")";

  this->constraints.distribute(increment);
  this->solution.block(0) += increment;
  this->solution.block(0).update_ghost_values();
}



// @sect4{LevelSetOKZMatrixSolver::compute_normal}
template <int dim>
void
adaflo::LevelSetOKZMatrixSolver<dim>::compute_normal(const bool fast_computation)
{
  // This function computes the normal from a projection of $\nabla C$ onto
  // the space of linear finite elements (with some small damping)

  if (fast_computation == true && normal_calculated &&
      this->parameters.approximate_projections == false)
    return;

  Assert(this->parameters.approximate_projections == false, ExcNotImplemented());

  TimerOutput::Scope timer(*this->timer, "Compute normal.");

  // compute right hand side
  this->normal_vector_rhs = 0;
  system_matrix           = 0;

  FEValues<dim> fe_values(this->mapping,
                          *this->fe,
                          this->matrix_free.get_quadrature(2),
                          update_gradients | update_values | update_quadrature_points |
                            update_JxW_values);

  const unsigned int dofs_per_cell = this->fe->dofs_per_cell;
  const unsigned int n_q_points    = fe_values.get_quadrature().size();

  FullMatrix<double>          cell_matrix(dofs_per_cell, dofs_per_cell);
  std::vector<Vector<double>> cell_rhs(dim, Vector<double>(dofs_per_cell));

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  std::vector<Tensor<1, dim>> ls_gradient(n_q_points);

  const double damping =
    (this->parameters.use_anisotropic_refinement == true ? 16. : 4.) *
    numbers::NumberTraits<double>::abs_square(this->epsilon_used /
                                              this->parameters.epsilon);


  typename DoFHandler<dim>::active_cell_iterator cell = this->dof_handler.begin_active(),
                                                 endc = this->dof_handler.end();
  for (; cell != endc; ++cell)
    if (cell->is_locally_owned())
      {
        for (unsigned int d = 0; d < dim; ++d)
          cell_rhs[d] = 0;

        fe_values.reinit(cell);
        if (fe_values.get_cell_similarity() != CellSimilarity::translation)
          cell_matrix = 0;
        fe_values.get_function_gradients(this->solution.block(0), ls_gradient);

        for (unsigned int q = 0; q < n_q_points; ++q)
          {
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                if (fe_values.get_cell_similarity() != CellSimilarity::translation)
                  for (unsigned int j = 0; j < dofs_per_cell; ++j)
                    {
                      cell_matrix(i, j) +=
                        (fe_values.shape_value(i, q) * fe_values.shape_value(j, q) +
                         fe_values.shape_grad(i, q) * fe_values.shape_grad(j, q) *
                           damping) *
                        fe_values.JxW(q);
                    }

                for (unsigned int d = 0; d < dim; ++d)
                  cell_rhs[d](i) +=
                    (fe_values.shape_value(i, q) * ls_gradient[q][d]) * fe_values.JxW(q);
              }
          }

        cell->get_dof_indices(local_dof_indices);
        this->constraints.distribute_local_to_global(cell_matrix,
                                                     local_dof_indices,
                                                     system_matrix);
        for (unsigned int d = 0; d < dim; ++d)
          this->constraints.distribute_local_to_global(cell_rhs[d],
                                                       local_dof_indices,
                                                       this->normal_vector_rhs.block(d));
      }
  this->normal_vector_rhs.compress(VectorOperation::add);
  system_matrix.compress(VectorOperation::add);

  TrilinosWrappers::PreconditionSSOR preconditioner;
  preconditioner.initialize(system_matrix);
  for (unsigned int d = 0; d < dim; ++d)
    {
      // solve linear system
      SolverControl solver_control(4000,
                                   1e-10 * this->normal_vector_rhs.block(d).l2_norm());

      SolverCG<LinearAlgebra::distributed::Vector<double>> cg(solver_control);
      cg.solve(system_matrix,
               this->normal_vector_field.block(d),
               this->normal_vector_rhs.block(d),
               preconditioner);

      this->constraints.distribute(this->normal_vector_field.block(d));
      this->normal_vector_field.block(d).update_ghost_values();
    }
  normal_calculated = true;
}



// @sect4{LevelSetOKZMatrixSolver::compute_normal}
template <int dim>
void
adaflo::LevelSetOKZMatrixSolver<dim>::compute_curvature(const bool)
{
  // This function computes the curvature from the normal field.
  compute_normal(false);

  TimerOutput::Scope timer(*this->timer, "Compute curvature.");

  // compute right hand side
  LinearAlgebra::distributed::Vector<double> &rhs = this->system_rhs.block(0);
  system_matrix                                   = 0;
  rhs                                             = 0;

  FEValues<dim> fe_values(this->mapping,
                          *this->fe,
                          this->matrix_free.get_quadrature(2),
                          update_gradients | update_values | update_quadrature_points |
                            update_JxW_values);

  const unsigned int dofs_per_cell = this->fe->dofs_per_cell;
  const unsigned int n_q_points    = fe_values.get_quadrature().size();

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  std::vector<Tensor<1, dim>> normals(n_q_points);

  const double damping =
    (this->parameters.use_anisotropic_refinement == true ? 4. : 0.5) *
    numbers::NumberTraits<double>::abs_square(this->epsilon_used /
                                              this->parameters.epsilon);

  typename DoFHandler<dim>::active_cell_iterator cell = this->dof_handler.begin_active(),
                                                 endc = this->dof_handler.end();
  for (; cell != endc; ++cell)
    if (cell->is_locally_owned())
      {
        fe_values.reinit(cell);
        if (fe_values.get_cell_similarity() != CellSimilarity::translation)
          cell_matrix = 0;
        for (unsigned int d = 0; d < dim; ++d)
          {
            cell->get_dof_values(this->normal_vector_field.block(d), cell_rhs);
            for (unsigned int q = 0; q < n_q_points; ++q)
              {
                double val = 0;
                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                  val += fe_values.shape_value(i, q) * cell_rhs(i);
                normals[q][d] = val;
              }
          }

        cell_rhs = 0;
        for (unsigned int q = 0; q < n_q_points; ++q)
          {
            Tensor<1, dim> normal = normals[q];
            if (normal.norm() > 1e-8)
              normal /= normal.norm();
            else
              normal *= 1e8;

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                if (fe_values.get_cell_similarity() != CellSimilarity::translation)
                  for (unsigned int j = 0; j < dofs_per_cell; ++j)
                    {
                      cell_matrix(i, j) +=
                        (fe_values.shape_value(i, q) * fe_values.shape_value(j, q) +
                         fe_values.shape_grad(i, q) * fe_values.shape_grad(j, q) *
                           damping) *
                        fe_values.JxW(q);
                    }

                cell_rhs(i) += ((fe_values.shape_grad(i, q) * normal)
                                // fe_values.shape_value(i,q) * rhs
                                // use this instead of first line for exact curvature
                                * fe_values.JxW(q));
              }
          }

        cell->get_dof_indices(local_dof_indices);
        this->constraints.distribute_local_to_global(cell_matrix,
                                                     local_dof_indices,
                                                     system_matrix);
        this->constraints.distribute_local_to_global(cell_rhs, local_dof_indices, rhs);
      }
  system_matrix.compress(VectorOperation::add);
  rhs.compress(VectorOperation::add);

  TrilinosWrappers::PreconditionSSOR preconditioner;
  preconditioner.initialize(system_matrix);
  SolverControl solver_control(1000, 1e-6 * rhs.l2_norm());
  SolverCG<LinearAlgebra::distributed::Vector<double>> cg(solver_control);
  cg.solve(system_matrix, this->solution.block(1), rhs, preconditioner);

  this->constraints.distribute(this->solution.block(1));

  // correct curvature away from the zero level set by computing the distance
  // and correcting the value, if so requested
  if (this->parameters.curvature_correction == true)
    {
      for (unsigned int i = 0; i < this->solution.block(1).locally_owned_size(); ++i)
        if (this->solution.block(1).local_element(i) > 1e-4)
          {
            const double c_val = this->solution.block(0).local_element(i);
            const double distance =
              (1 - c_val * c_val) > 1e-2 ?
                this->epsilon_used * std::log((1. + c_val) / (1. - c_val)) :
                0;

            this->solution.block(1).local_element(i) =
              1. / (1. / this->solution.block(1).local_element(i) + distance / (dim - 1));
          }
    }
  this->solution.block(1).update_ghost_values();
}



// force is only active in a few elements around the interface. find these
// elements now
template <int dim>
void
adaflo::LevelSetOKZMatrixSolver<dim>::compute_heaviside()
{
  TimerOutput::Scope timer(*this->timer, "Compute Heaviside.");
  const double       cutoff = std::tanh(2);

  const unsigned int                   dofs_per_cell = this->fe->dofs_per_cell;
  Vector<double>                       local_values(dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  typename DoFHandler<dim>::active_cell_iterator cell = this->dof_handler.begin_active(),
                                                 endc = this->dof_handler.end();
  for (; cell != endc; ++cell)
    if (cell->is_locally_owned()) // Check if the cell is owned by the local
      // processor
      {
        cell->get_dof_indices(local_dof_indices);
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          local_values[i] = this->solution.block(0)[local_dof_indices[i]];
        bool consider_cell = false;
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          if (std::fabs(local_values(i)) < cutoff)
            {
              consider_cell = true;
              break;
            }
        if (consider_cell == true)
          {
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                const double c_val    = local_values(i);
                double       distance = 0;
                if (c_val < -cutoff)
                  distance = -3;
                else if (c_val > cutoff)
                  distance = 3;
                else
                  distance = std::log((1 + c_val) / (1 - c_val));

                // want to have force along width 2*h which is
                // what distance scaled by twice relative
                // epsilon but not mesh size is doing in
                // discrete_heaviside that is defined between
                // -2 and 2.
                distance *= this->parameters.epsilon * 2. /
                            this->parameters.concentration_subdivisions;
                this->heaviside(local_dof_indices[i]) = discrete_heaviside(distance);
              }
          }
        else
          {
            if (local_values(0) < 0)
              for (unsigned int i = 0; i < dofs_per_cell; ++i)
                this->heaviside(local_dof_indices[i]) = 0.;
            else
              for (unsigned int i = 0; i < dofs_per_cell; ++i)
                this->heaviside(local_dof_indices[i]) = 1.;
          }
      }
  this->heaviside.update_ghost_values();
}



template <int dim>
void
adaflo::LevelSetOKZMatrixSolver<dim>::reinitialize(const unsigned int stab_steps,
                                                   const unsigned int diff_steps,
                                                   const bool)
{
  TimerOutput::Scope timer(*this->timer, "Reinitialization.");

  // This function assembles and solves for a given profile using the approach
  // described in the paper by Olsson, Kreiss, and Zahedi.

  std::cout.precision(3);

  // perform several reinitialization steps until we reach the maximum number
  // of steps. TODO: make an adaptive choice of the number of iterations
  if (!this->parameters.do_iteration)
    this->pcout << (this->time_stepping.now() == this->time_stepping.start() ? "  " :
                                                                               " and ")
                << "reinitialize (";

  bool diffuse_only = false;

  const double dtau = 0.3 * std::min(this->epsilon_used / this->parameters.epsilon,
                                     this->time_stepping.step_size());

  for (unsigned int tau = 0; tau < diff_steps + stab_steps; tau++)
    {
      if (tau >= diff_steps && (tau - diff_steps) % 8 == 0)
        compute_normal(true);

      if (tau < diff_steps)
        diffuse_only = true;

      // compute right hand side
      LinearAlgebra::distributed::Vector<double> &rhs = this->system_rhs.block(0);
      LinearAlgebra::distributed::Vector<double> &increment =
        this->solution_update.block(0);
      rhs           = 0;
      system_matrix = 0;

      FEValues<dim> fe_values(this->mapping,
                              *this->fe,
                              this->matrix_free.get_quadrature(2),
                              update_values | update_gradients |
                                update_quadrature_points | update_JxW_values);

      const unsigned int dofs_per_cell = this->fe->dofs_per_cell;
      const unsigned int n_q_points    = fe_values.get_quadrature().size();

      FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
      Vector<double>     cell_rhs(dofs_per_cell);

      std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

      std::vector<double>         cur_solution_values(n_q_points);
      std::vector<Tensor<1, dim>> cur_solution_gradients(n_q_points);
      std::vector<Tensor<1, dim>> normal_values(n_q_points);

      for (typename DoFHandler<dim>::active_cell_iterator cell =
             this->dof_handler.begin_active();
           cell != this->dof_handler.end();
           ++cell)
        if (cell->is_locally_owned())
          {
            cell_matrix = 0;

            fe_values.reinit(cell);

            const double cell_diameter = cell->diameter() / std::sqrt(dim);
            double       diffusion     = std::max(this->epsilon_used, cell_diameter);

            fe_values.get_function_values(this->solution.block(0), cur_solution_values);
            fe_values.get_function_gradients(this->solution.block(0),
                                             cur_solution_gradients);

            // disable the compressive flux in case the function is too close
            // to a constant within the element

            bool disable_convection = false;
            if (!diffuse_only)
              {
                for (unsigned int d = 0; d < dim; ++d)
                  {
                    cell->get_dof_values(this->normal_vector_field.block(d), cell_rhs);
                    for (unsigned int q = 0; q < n_q_points; ++q)
                      {
                        double val = 0;
                        for (unsigned int i = 0; i < dofs_per_cell; ++i)
                          val += fe_values.shape_value(i, q) * cell_rhs(i);
                        normal_values[q][d] = val;
                      }
                  }

                double norm = 0;
                for (unsigned int q = 0; q < n_q_points; ++q)
                  norm += std::fabs(cur_solution_values[q] * cur_solution_values[q] - 1.);
                if (norm < 1e-10)
                  disable_convection = true;
              }

            cell_rhs = 0;
            // actual assembly loop
            for (unsigned int q = 0; q < n_q_points; ++q)
              {
                // use normal from the projected field. Using the gradient of
                // solution is less accurate (higher spurious velocities)

                Tensor<1, dim> normal =
                  diffuse_only ? Tensor<1, dim>() : normal_values[q];
                if (!disable_convection && !diffuse_only)
                  normal /= normal.norm();

                const double diff_val = diffusion * normal * cur_solution_gradients[q];
                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                  {
                    // standard reinit: include diffusion and mass matrix term
                    // in matrix, and put the compressive flux on the right
                    // hand side (explicit time stepping). just compute an
                    // update to the previous time level and to not include
                    // the previous solution. this allows measuring the
                    // residual

                    if (!diffuse_only && !disable_convection)
                      {
                        double n_nabla_eta = normal * fe_values.shape_grad(i, q);

                        for (unsigned int j = 0; j < dofs_per_cell; ++j)
                          cell_matrix(i, j) += (fe_values.shape_value(i, q) / dtau *
                                                  fe_values.shape_value(j, q) +
                                                n_nabla_eta * diffusion *
                                                  (normal * fe_values.shape_grad(j, q))) *
                                               fe_values.JxW(q);

                        cell_rhs(i) +=
                          (n_nabla_eta *
                           (0.5 * (1. - cur_solution_values[q] * cur_solution_values[q]) -
                            diff_val)) *
                          fe_values.JxW(q);
                      }
                    // just diffusion
                    else
                      {
                        for (unsigned int j = 0; j < dofs_per_cell; ++j)
                          cell_matrix(i, j) += (fe_values.shape_value(i, q) / dtau *
                                                  fe_values.shape_value(j, q) +
                                                diffusion * fe_values.shape_grad(i, q) *
                                                  fe_values.shape_grad(j, q)) *
                                               fe_values.JxW(q);

                        cell_rhs(i) -=
                          (fe_values.shape_grad(i, q) * cur_solution_gradients[q]) *
                          diffusion * fe_values.JxW(q);
                      }
                  }
              }

            cell->get_dof_indices(local_dof_indices);
            this->constraints.distribute_local_to_global(cell_matrix,
                                                         local_dof_indices,
                                                         system_matrix);
            this->constraints.distribute_local_to_global(cell_rhs,
                                                         local_dof_indices,
                                                         rhs);
          }

      rhs.compress(VectorOperation::add);
      system_matrix.compress(VectorOperation::add);


      // solve linear system
      TrilinosWrappers::PreconditionSSOR preconditioner;
      preconditioner.initialize(system_matrix);
      increment = 0;
      SolverControl                                        solver_control(1000,
                                   std::max(0.02 * this->parameters.tol_nl_iteration,
                                            1e-6 * rhs.l2_norm()));
      SolverCG<LinearAlgebra::distributed::Vector<double>> cg(solver_control);
      cg.solve(system_matrix, increment, rhs, preconditioner);

      this->constraints.distribute(increment);
      if (!this->parameters.do_iteration)
        {
          if (tau < diff_steps)
            this->pcout << "d" << solver_control.last_step();
          else
            this->pcout << solver_control.last_step();
        }

      this->solution.block(0) += increment;
      this->solution.block(0).update_ghost_values();

      // check residual
      const double update_norm = increment.l2_norm();
      if (update_norm < 1e-6)
        break;

      normal_calculated = false;
      if (!this->parameters.do_iteration && tau < diff_steps + stab_steps - 1)
        this->pcout << " + ";
    }

  if (!this->parameters.do_iteration)
    this->pcout << ")" << std::endl << std::flush;
}



template class adaflo::LevelSetOKZMatrixSolver<2>;
template class adaflo::LevelSetOKZMatrixSolver<3>;
