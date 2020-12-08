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
#include <deal.II/base/vectorization.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_q_iso_q1.h>

#include <deal.II/grid/tria.h>

#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_bicgstab.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_richardson.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

#include <deal.II/numerics/vector_tools.h>

#include <adaflo/block_matrix_extension.h>
#include <adaflo/level_set_okz.h>
#include <adaflo/level_set_okz_preconditioner.h>
#include <adaflo/level_set_okz_template_instantations.h>
#include <adaflo/util.h>

#include <fstream>
#include <iostream>


using namespace dealii;



template <int dim>
LevelSetOKZSolver<dim>::LevelSetOKZSolver(const FlowParameters &parameters_in,
                                          Triangulation<dim> &  tria_in)
  : LevelSetBaseAlgorithm<dim>(parameters_in, tria_in)
  , first_reinit_step(true)
{
  {
    LevelSetOKZSolverComputeNormalParameter params;
    params.dof_index_ls               = 2;
    params.dof_index_normal           = 4;
    params.quad_index                 = 2;
    params.concentration_subdivisions = this->parameters.concentration_subdivisions;
    params.epsilon                    = this->parameters.epsilon;
    params.approximate_projections    = this->parameters.approximate_projections;

    this->normal_operator =
      std::make_unique<LevelSetOKZSolverComputeNormal<dim>>(this->normal_vector_field,
                                                            this->normal_vector_rhs,
                                                            this->solution.block(0),
                                                            this->cell_diameters,
                                                            this->epsilon_used,
                                                            this->minimal_edge_length,
                                                            this->constraints_normals,
                                                            params,
                                                            this->matrix_free,
                                                            this->preconditioner,
                                                            this->projection_matrix,
                                                            this->ilu_projection_matrix);
  }

  {
    LevelSetOKZSolverReinitializationParameter params;

    params.dof_index_ls               = 2;
    params.dof_index_normal           = 4;
    params.quad_index                 = 2;
    params.concentration_subdivisions = this->parameters.concentration_subdivisions;
    params.do_iteration               = this->parameters.do_iteration;

    // set time stepping parameters of level set to correspond with the values from
    // Navier-Stokes
    // @todo
    params.time.time_step_scheme     = this->parameters.time_step_scheme;
    params.time.start_time           = this->parameters.start_time;
    params.time.end_time             = this->parameters.end_time;
    params.time.time_step_size_start = this->parameters.time_step_size_start;
    params.time.time_stepping_cfl    = this->parameters.time_stepping_cfl;
    params.time.time_stepping_coef2  = this->parameters.time_stepping_coef2;
    params.time.time_step_tolerance  = this->parameters.time_step_tolerance;
    params.time.time_step_size_max   = this->parameters.time_step_size_max;
    params.time.time_step_size_min   = this->parameters.time_step_size_min;

    this->reinit_operator = std::make_unique<LevelSetOKZSolverReinitialization<dim>>(
      *this->normal_operator,
      this->normal_vector_field,
      this->cell_diameters,
      this->epsilon_used,
      this->minimal_edge_length,
      this->constraints,
      this->solution_update.block(0),
      this->solution.block(0),
      this->system_rhs.block(0),
      this->pcout,
      this->preconditioner,
      this->last_concentration_range,
      params,
      this->first_reinit_step,
      this->matrix_free);
  }

  {
    LevelSetOKZSolverComputeCurvatureParameter params;
    params.dof_index_curvature        = 3;
    params.dof_index_normal           = 4;
    params.quad_index                 = 2;
    params.concentration_subdivisions = this->parameters.concentration_subdivisions;
    params.epsilon                    = this->parameters.epsilon;
    params.approximate_projections    = this->parameters.approximate_projections;
    params.curvature_correction       = this->parameters.curvature_correction;

    this->curvatur_operator = std::make_unique<LevelSetOKZSolverComputeCurvature<dim>>(
      *this->normal_operator,
      this->cell_diameters,
      this->normal_vector_field,
      this->constraints_curvature,
      this->constraints,
      this->epsilon_used,
      this->system_rhs.block(0),
      params,
      this->solution.block(1),
      this->solution.block(0),
      this->matrix_free,
      preconditioner,
      projection_matrix,
      ilu_projection_matrix);
  }

  // set up advection operator
  {
    LevelSetOKZSolverAdvanceConcentrationParameter params;

    params.dof_index_ls               = 2;
    params.dof_index_vel              = 0;
    params.quad_index                 = 2;
    params.concentration_subdivisions = this->parameters.concentration_subdivisions;
    params.convection_stabilization   = this->parameters.convection_stabilization;
    params.do_iteration               = this->parameters.do_iteration;
    params.tol_nl_iteration           = this->parameters.tol_nl_iteration;

    LevelSetOKZSolverAdvanceConcentrationBoundaryDescriptor<dim> bcs;

    bcs.dirichlet = this->boundary->fluid_type;
    bcs.symmetry  = this->boundary->symmetry;

    // set time stepping parameters of level set to correspond with the values from
    // Navier-Stokes
    // @todo
    params.time.time_step_scheme     = this->parameters.time_step_scheme;
    params.time.start_time           = this->parameters.start_time;
    params.time.end_time             = this->parameters.end_time;
    params.time.time_step_size_start = this->parameters.time_step_size_start;
    params.time.time_stepping_cfl    = this->parameters.time_stepping_cfl;
    params.time.time_stepping_coef2  = this->parameters.time_stepping_coef2;
    params.time.time_step_tolerance  = this->parameters.time_step_tolerance;
    params.time.time_step_size_max   = this->parameters.time_step_size_max;
    params.time.time_step_size_min   = this->parameters.time_step_size_min;

    this->advection_operator =
      std::make_unique<LevelSetOKZSolverAdvanceConcentration<dim>>(
        this->solution.block(0),
        this->solution_old.block(0),
        this->solution_old_old.block(0),
        this->solution_update.block(0),
        this->system_rhs.block(0),
        this->navier_stokes.solution.block(0),
        this->navier_stokes.solution_old.block(0),
        this->navier_stokes.solution_old_old.block(0),
        this->global_omega_diameter,
        this->cell_diameters,
        this->constraints,
        this->pcout,
        bcs,
        this->matrix_free,
        params,
        this->global_max_velocity,
        this->preconditioner);
  }
}



template <int dim>
void
LevelSetOKZSolver<dim>::transform_distance_function(
  LinearAlgebra::distributed::Vector<double> &vector) const
{
  Assert(this->epsilon_used > 0, ExcInternalError());
  for (unsigned int i = 0; i < vector.local_size(); i++)
    vector.local_element(i) =
      -std::tanh(vector.local_element(i) / (2. * this->epsilon_used));
}


// @sect4{LevelSetOKZSolver::make_grid_and_dofs}
template <int dim>
void
LevelSetOKZSolver<dim>::initialize_data_structures()
{
  this->LevelSetBaseAlgorithm<dim>::initialize_data_structures();

  initialize_mass_matrix_diagonal(
    this->matrix_free, this->hanging_node_constraints, 2, 2, preconditioner);

  projection_matrix     = std::make_shared<BlockMatrixExtension>();
  ilu_projection_matrix = std::make_shared<BlockILUExtension>();

  initialize_projection_matrix(this->matrix_free,
                               this->constraints_normals,
                               2,
                               2,
                               this->epsilon_used,
                               this->parameters.epsilon,
                               this->cell_diameters,
                               *projection_matrix,
                               *ilu_projection_matrix);
}



template <int dim>
void
LevelSetOKZSolver<dim>::local_projection_matrix(
  const MatrixFree<dim> &                                           data,
  std::shared_ptr<Threads::ThreadLocalStorage<AssemblyData::Data>> &scratch,
  const unsigned int &,
  const std::pair<unsigned int, unsigned int> &cell_range)
{
  const unsigned int ls_degree = this->parameters.concentration_subdivisions;
  if (ls_degree == 1)
    local_projection_matrix<1>(data, scratch, cell_range);
  else if (ls_degree == 2)
    local_projection_matrix<2>(data, scratch, cell_range);
  else if (ls_degree == 3)
    local_projection_matrix<3>(data, scratch, cell_range);
  else if (ls_degree == 4)
    local_projection_matrix<4>(data, scratch, cell_range);
  else
    AssertThrow(false, ExcNotImplemented());
}



template <int dim>
template <int ls_degree>
void
LevelSetOKZSolver<dim>::local_projection_matrix(
  const MatrixFree<dim> &                                           data,
  std::shared_ptr<Threads::ThreadLocalStorage<AssemblyData::Data>> &scratch_data,
  const std::pair<unsigned int, unsigned int> &                     cell_range)
{
  FEEvaluation<dim, ls_degree, 2 * ls_degree, 1, double> phi(data, 4, 2);
  AssemblyData::Data &                                   scratch = scratch_data->get();

  const VectorizedArray<double> min_diameter =
    make_vectorized_array(this->epsilon_used / this->parameters.epsilon);

  for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      phi.reinit(cell);
      const VectorizedArray<double> damping =
        4. * Utilities::fixed_power<2>(
               std::max(min_diameter,
                        this->cell_diameters[cell] / static_cast<double>(ls_degree)));

      for (unsigned int i = 0; i < phi.dofs_per_cell; ++i)
        {
          for (unsigned int j = 0; j < phi.dofs_per_cell; ++j)
            phi.begin_dof_values()[j] = VectorizedArray<double>();
          phi.begin_dof_values()[i] = 1.;
          phi.evaluate(true, true);
          for (unsigned int q = 0; q < phi.n_q_points; ++q)
            {
              phi.submit_value(phi.get_value(q), q);
              phi.submit_gradient(phi.get_gradient(q) * damping, q);
            }
          phi.integrate(true, true);
          for (unsigned int v = 0; v < data.n_active_entries_per_cell_batch(cell); ++v)
            for (unsigned int j = 0; j < phi.dofs_per_cell; ++j)
              scratch.matrices[v](phi.get_shape_info().lexicographic_numbering[j],
                                  phi.get_shape_info().lexicographic_numbering[i]) =
                phi.begin_dof_values()[j][v];
        }
      for (unsigned int v = 0; v < data.n_active_entries_per_cell_batch(cell); ++v)
        {
          typename DoFHandler<dim>::active_cell_iterator dcell =
            this->matrix_free.get_cell_iterator(cell, v, 2);
          dcell->get_dof_indices(scratch.dof_indices);
          this->constraints_normals.distribute_local_to_global(
            scratch.matrices[v],
            scratch.dof_indices,
            static_cast<TrilinosWrappers::SparseMatrix &>(*projection_matrix));
        }
    }
}



template <int dim>
template <int ls_degree, int velocity_degree>
void
LevelSetOKZSolver<dim>::local_compute_force(
  const MatrixFree<dim, double> &             data,
  LinearAlgebra::distributed::Vector<double> &dst,
  const LinearAlgebra::distributed::Vector<double> &,
  const std::pair<unsigned int, unsigned int> &cell_range)
{
  // The second input argument below refers to which constrains should be used,
  // 2 means constraints (for LS-function), 3 means constraints_curvature and so
  // an
  FEEvaluation<dim, ls_degree, velocity_degree + 1, 1>           ls_values(data, 2, 0);
  FEEvaluation<dim, ls_degree, velocity_degree + 1, 1>           curv_values(data, 3, 0);
  FEEvaluation<dim, velocity_degree, velocity_degree + 1, dim>   vel_values(data, 0, 0);
  FEEvaluation<dim, velocity_degree - 1, velocity_degree + 1, 1> pre_values(data, 1, 0);

  typedef VectorizedArray<double> vector_t;
  const bool                      use_variable_parameters =
    this->parameters.density_diff != 0 || this->parameters.viscosity_diff != 0;

  for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      ls_values.reinit(cell);
      vel_values.reinit(cell);
      curv_values.reinit(cell);
      pre_values.reinit(cell);

      ls_values.read_dof_values_plain(this->heaviside);

      // set variable parameters
      if (use_variable_parameters)
        {
          vector_t *densities = this->navier_stokes.get_matrix().begin_densities(cell);
          vector_t *viscosities =
            this->navier_stokes.get_matrix().begin_viscosities(cell);
          ls_values.evaluate(true, false);
          for (unsigned int q = 0; q < ls_values.n_q_points; ++q)
            {
              densities[q] = this->parameters.density +
                             this->parameters.density_diff * ls_values.get_value(q);
              viscosities[q] = this->parameters.viscosity +
                               this->parameters.viscosity_diff * ls_values.get_value(q);
            }
        }

      // interpolate ls values onto pressure
      if (this->parameters.interpolate_grad_onto_pressure)
        {
          for (unsigned int i = 0; i < pre_values.dofs_per_cell; ++i)
            {
              vector_t projected_value = vector_t();
              for (unsigned int j = 0; j < ls_values.dofs_per_cell; ++j)
                projected_value += this->interpolation_concentration_pressure(i, j) *
                                   ls_values.get_dof_value(j);
              pre_values.submit_dof_value(projected_value, i);
            }
          pre_values.evaluate(false, true);
        }
      else
        ls_values.evaluate(false, true);

      // evaluate curvature and level set gradient
      curv_values.read_dof_values_plain(this->solution.block(1));
      curv_values.evaluate(true, false);

      // evaluate surface tension force and gravity force
      for (unsigned int q = 0; q < curv_values.n_q_points; ++q)
        {
          // surface tension
          Tensor<1, dim, vector_t> force =
            (this->parameters.surface_tension * curv_values.get_value(q)) *
            (this->parameters.interpolate_grad_onto_pressure ?
               pre_values.get_gradient(q) :
               ls_values.get_gradient(q));

          // gravity
          vector_t actual_rho =
            use_variable_parameters ?
              *(this->navier_stokes.get_matrix().begin_densities(cell) + q) :
              make_vectorized_array(this->parameters.density);
          force[dim - 1] -= this->parameters.gravity * actual_rho;

          vel_values.submit_value(force, q);
        }
      vel_values.integrate(true, false);

      vel_values.distribute_local_to_global(dst);
    }
}

template <int dim>
void
LevelSetOKZSolver<dim>::compute_force()
{
  compute_heaviside();
  compute_curvature();

  TimerOutput::Scope timer(*this->timer, "LS compute force.");

  this->compute_density_on_faces();

  this->navier_stokes.user_rhs = 0;
#define OPERATION(ls_degree, vel_degree)                                          \
  this->matrix_free.cell_loop(                                                    \
    &LevelSetOKZSolver<dim>::template local_compute_force<ls_degree, vel_degree>, \
    this,                                                                         \
    this->navier_stokes.user_rhs.block(0),                                        \
    this->solution.block(0))

  EXPAND_OPERATIONS(OPERATION);
#undef OPERATION
}



// @sect4{LevelSetOKZSolver::advance_concentration}
template <int dim>
void
LevelSetOKZSolver<dim>::advance_concentration()
{
  TimerOutput::Scope timer(*this->timer, "LS advance concentration.");
  advection_operator->advance_concentration(this->time_stepping.step_size());
}



// @sect4{LevelSetOKZSolver::compute_normal}
template <int dim>
void
LevelSetOKZSolver<dim>::compute_normal(const bool fast_computation)
{
  TimerOutput::Scope timer(*this->timer, "LS compute normal.");
  normal_operator->compute_normal(fast_computation);
}



// @sect4{LevelSetOKZSolver::compute_normal}
template <int dim>
void
LevelSetOKZSolver<dim>::compute_curvature(const bool diffuse_large_values)
{
  TimerOutput::Scope timer(*this->timer, "LS compute curvature.");
  curvatur_operator->compute_curvature(diffuse_large_values);
}



// force is only active in a few elements around the interface. find these
// elements now
template <int dim>
void
LevelSetOKZSolver<dim>::compute_heaviside()
{
  TimerOutput::Scope timer(*this->timer, "LS compute Heaviside.");
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
LevelSetOKZSolver<dim>::reinitialize(const unsigned int stab_steps,
                                     const unsigned int diff_steps,
                                     const bool diffuse_cells_with_large_curvature_only)
{
  TimerOutput::Scope timer(*this->timer, "LS reinitialization step.");
  reinit_operator->reinitialize(this->time_stepping.step_size(),
                                stab_steps,
                                diff_steps,
                                diffuse_cells_with_large_curvature_only);
}



template class LevelSetOKZSolver<2>;
template class LevelSetOKZSolver<3>;
