// --------------------------------------------------------------------------
//
// Copyright (C) 2011 - 2016 by the adaflo authors
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
#include <deal.II/fe/fe_q_dg0.h>
#include <deal.II/fe/fe_q_iso_q1.h>
#include <deal.II/fe/fe_simplex_p.h>

#include <deal.II/grid/tria.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/solver_bicgstab.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparsity_tools.h>

#include <deal.II/numerics/vector_tools.h>

#include <adaflo/level_set_base.h>
#include <adaflo/util.h>

#include <fstream>
#include <iostream>


using namespace dealii;


template <int dim>
adaflo::LevelSetBaseAlgorithm<dim>::LevelSetBaseAlgorithm(
  const FlowParameters &parameters_in,
  Triangulation<dim>   &tria_in)
  : TwoPhaseBaseAlgorithm<dim>(parameters_in,
                               parameters_in.use_simplex_mesh ?
                                 std::shared_ptr<FiniteElement<dim>>(new FE_SimplexP<dim>(
                                   parameters_in.concentration_subdivisions)) :
                                 std::shared_ptr<FiniteElement<dim>>(new FE_Q_iso_Q1<dim>(
                                   parameters_in.concentration_subdivisions)),
                               tria_in)
  , old_residual(std::numeric_limits<double>::max())
  , last_smoothing_step(0)
  , last_refine_step(0)
{
  // computes the interpolation matrix from level set functions to pressure
  // which is needed for evaluating surface tension
  {
    interpolation_concentration_pressure.reinit(
      this->navier_stokes.get_fe_p().dofs_per_cell, this->fe->dofs_per_cell);
    if (this->parameters.use_simplex_mesh)
      {
        const auto fe_mine = dynamic_cast<const FE_SimplexP<dim> *>(this->fe.get());
        const auto fe_p =
          dynamic_cast<const FE_SimplexP<dim> *>(&this->navier_stokes.get_fe_p());

        Assert(fe_mine, ExcNotImplemented());
        Assert(fe_p, ExcNotImplemented());

        /*
        const std::vector<unsigned int> lexicographic_ls =
          fe_mine->get_poly_space_numbering_inverse();

        const std::vector<unsigned int> lexicographic_p =
          fe_p->get_poly_space_numbering_inverse();
        */

        std::vector<unsigned int> lexicographic_ls(fe_mine->dofs_per_cell);
        std::vector<unsigned int> lexicographic_p(fe_p->dofs_per_cell);

        for (unsigned int j = 0; j < fe_mine->dofs_per_cell; ++j)
          lexicographic_ls[j] = j;

        for (unsigned int j = 0; j < fe_p->dofs_per_cell; ++j)
          lexicographic_p[j] = j;

        for (unsigned int j = 0; j < fe_p->dofs_per_cell; ++j)
          {
            const Point<dim> p = fe_p->get_unit_support_points()[lexicographic_p[j]];
            for (unsigned int i = 0; i < fe_mine->dofs_per_cell; ++i)
              interpolation_concentration_pressure(j, i) =
                this->fe->shape_value(lexicographic_ls[i], p);
          }
      }
    else
      {
        const FE_Q_iso_Q1<dim> &fe_mine =
          dynamic_cast<const FE_Q_iso_Q1<dim> &>(*this->fe);
        const std::vector<unsigned int> lexicographic_ls =
          fe_mine.get_poly_space_numbering_inverse();
        if (const FE_Q<dim> *fe_p =
              dynamic_cast<const FE_Q<dim> *>(&this->navier_stokes.get_fe_p()))
          {
            const std::vector<unsigned int> lexicographic_p =
              fe_p->get_poly_space_numbering_inverse();
            for (unsigned int j = 0; j < fe_p->dofs_per_cell; ++j)
              {
                const Point<dim> p = fe_p->get_unit_support_points()[lexicographic_p[j]];
                for (unsigned int i = 0; i < fe_mine.dofs_per_cell; ++i)
                  interpolation_concentration_pressure(j, i) =
                    this->fe->shape_value(lexicographic_ls[i], p);
              }
          }
        else if (const FE_Q_DG0<dim> *fe_p =
                   dynamic_cast<const FE_Q_DG0<dim> *>(&this->navier_stokes.get_fe_p()))
          {
            const std::vector<unsigned int> lexicographic_p =
              fe_p->get_poly_space_numbering_inverse();
            for (unsigned int j = 0; j < fe_p->dofs_per_cell - 1; ++j)
              {
                const Point<dim> p = fe_p->get_unit_support_points()[lexicographic_p[j]];
                for (unsigned int i = 0; i < fe_mine.dofs_per_cell; ++i)
                  interpolation_concentration_pressure(j, i) =
                    this->fe->shape_value(lexicographic_ls[i], p);
              }
          }
      }
  }

  this->curvature_name = "curvature";
}



template <int dim>
void
adaflo::LevelSetBaseAlgorithm<dim>::setup_problem(
  const Function<dim> &initial_velocity_field,
  const Function<dim> &initial_distance_function)
{
  this->TwoPhaseBaseAlgorithm<dim>::setup_problem(initial_velocity_field,
                                                  initial_distance_function);
  reinitialize(this->parameters.n_initial_reinit_steps);
  this->compute_heaviside();
}



// @sect4{LevelSetBaseAlgorithm::make_grid_and_dofs}
template <int dim>
void
adaflo::LevelSetBaseAlgorithm<dim>::initialize_data_structures()
{
  // now to the boundary conditions: the matrix system gets zero boundary
  // conditions on open boundaries
  Functions::ZeroFunction<dim>                        zero_func(1);
  std::map<types::boundary_id, const Function<dim> *> homogeneous_dirichlet;
  for (const auto &it : this->boundary->fluid_type)
    homogeneous_dirichlet[it.first] = &zero_func;
  VectorTools::interpolate_boundary_values(this->dof_handler,
                                           homogeneous_dirichlet,
                                           this->constraints);

  this->TwoPhaseBaseAlgorithm<dim>::initialize_data_structures();

  // resize the solution and right hand side vectors

  heaviside.reinit(this->solution.block(0));

  this->normal_vector_field.reinit(dim);
  for (unsigned int d = 0; d < dim; ++d)
    this->normal_vector_field.block(d).reinit(this->solution.block(0));
  this->normal_vector_field.collect_sizes();
  normal_vector_rhs.reinit(this->normal_vector_field);
}



template <int dim>
std::pair<unsigned int, unsigned int>
adaflo::LevelSetBaseAlgorithm<dim>::advance_time_step()
{
  // advance the time in the time stepping scheme. The Navier--Stokes class
  // takes care of setting the time step size and computing the current
  // time.
  this->init_time_advance();

  // this is the Newton iteration from the Navier-Stokes equations, plus a
  // Gauss-Seidel iteration for handling the nonlinearity of the concentration
  if (this->parameters.do_iteration)
    {
      AssertThrow(false, ExcNotImplemented());
      unsigned int step = 0;

      // This code is not tested and does probably not work
      /*

      this->pcout << "  Iterations/residual: ";

      std::vector<LinearAlgebra::distributed::BlockVector<double>*> old_vectors;
      old_vectors.push_back (&this->navier_stokes.solution_old);
      old_vectors.push_back (&this->navier_stokes.solution_old_old);
      this->navier_stokes.user_rhs = 0;
      this->navier_stokes.get_matrix().old_times (this->navier_stokes.const_rhs,
      old_vectors); for ( ; step<20; ++step)
        {
          this->advance_concentration();
          this->reinitialize (this->parameters.n_reinit_steps);
          this->compute_force();
          this->navier_stokes.system_rhs += this->navier_stokes.const_rhs;
          this->navier_stokes.get_matrix().residual
      (this->navier_stokes.system_rhs, this->navier_stokes.solution,
                                                     this->navier_stokes.user_rhs);
          const double res = this->navier_stokes.system_rhs.l2_norm();

          this->pcout << "[" << res;

          if (res < 1e-10)
            {
              this->pcout << "/conv.]" << std::endl;
              break;
            }

          if (this->navier_stokes.get_update_preconditioner()==true)
            this->navier_stokes.build_preconditioner();

          const std::pair<unsigned int,double> convergence =
            this->navier_stokes.solve_system (1e-3*res);
          this->navier_stokes.solution += this->navier_stokes.solution_update;

          this->navier_stokes.solution.update_ghost_values();

          this->pcout << "/" << convergence.first << "] " << std::flush;
        }
      */
      return {step, 0};
    }
  // do not iterate, just do an extrapolated value on the concentration
  else
    {
      // TODO: allow the user to choose between first advancing Navier-Stokes
      // and then level set and the other way around through the parameter
      // file
      this->advance_concentration();
      this->reinitialize(this->parameters.n_reinit_steps);
      this->compute_force();


      // When the curvature gets bad, the initial Navier-Stokes residual
      // becomes large. We should detect this case. One idea would be to let
      // Navier-Stokes compute the residual before we do the nonlinear
      // solution and check its size compared to previous steps. If it became
      // more than a factor 3 larger, we should try with some diffusion.

      // compute part of the residual from time level n
      double actual_res = this->navier_stokes.compute_initial_residual();

      if (this->time_stepping.step_no() > 3 + last_smoothing_step &&
          this->time_stepping.step_no() > 2 + last_refine_step &&
          actual_res >= 2. * old_residual)
        {
          this->pcout << std::endl << "  Correct excessive residual: ";
          // Add additional diffusion steps before starting to solve
          this->reinitialize(this->parameters.n_reinit_steps, 10);
          this->compute_force();
          actual_res          = this->navier_stokes.compute_initial_residual();
          last_smoothing_step = this->time_stepping.step_no();
        }
      old_residual = actual_res;

      std::pair<unsigned int, unsigned int> iter({0, 0});
      try
        {
          iter = this->navier_stokes.solve_nonlinear_system(actual_res);
        }
      catch (const ExcNavierStokesNoConvergence &e)
        {
          this->pcout << "Warning: nonlinear iteration did not converge!" << std::endl;
        }
      return iter;
    }
}



template <int dim>
bool
adaflo::LevelSetBaseAlgorithm<dim>::mark_cells_for_refinement()
{
  if (this->parameters.adaptive_refinements == 0)
    return false;

  if (this->time_stepping.step_no() == 0)
    compute_normal(true);

  TimerOutput::Scope timer(*this->timer, "Probe grid refinement.");

  const int upper_level_limit =
    this->parameters.adaptive_refinements + this->refine_lower_level_limit;

  // look towards the end of the elements to find extrema in the error
  // indicator (= level set gradient)
  std::vector<Point<1>> point(2);
  point[0][0] = 0.05;
  point[1][0] = 0.95;
  Quadrature<1>   quadrature_1d(point);
  Quadrature<dim> quadrature(quadrature_1d);
  FEValues<dim>   fe_values(*this->fe, quadrature, update_values);
  FEValues<dim>   vel_values(this->navier_stokes.get_fe_u(), quadrature, update_values);
  std::vector<std::vector<double>> ls_gradients(dim,
                                                std::vector<double>(quadrature.size()));
  std::vector<double>              ls_values(quadrature.size());
  std::vector<Tensor<1, dim>>      u_values(quadrature.size());

  const FEValuesExtractors::Vector               velocity(0);
  typename DoFHandler<dim>::active_cell_iterator cell = this->dof_handler.begin_active(),
                                                 ns_cell =
                                                   this->navier_stokes.get_dof_handler_u()
                                                     .begin_active(),
                                                 endc = this->dof_handler.end();

  bool needs_refinement_or_coarsening = false;

  for (; cell != endc; ++cell)
    if (cell->is_locally_owned()) // Cell belongs to the local core or not
      {
        cell->clear_coarsen_flag();
        cell->clear_refine_flag();
        fe_values.reinit(cell);
        vel_values.reinit(ns_cell);
        for (unsigned int d = 0; d < dim; ++d)
          fe_values.get_function_values(this->normal_vector_field.block(d),
                                        ls_gradients[d]);
        double distance = 0;
        for (unsigned int q = 0; q < quadrature.size(); ++q)
          {
            Tensor<1, dim> ls_gradient;
            for (unsigned int d = 0; d < dim; ++d)
              ls_gradient[d] = ls_gradients[d][q];
            distance = std::max(distance, ls_gradient.norm());
          }

        distance = std::log(distance * this->epsilon_used);

        if ((cell->level() < upper_level_limit && distance > -3.5) ||
            (this->time_stepping.step_no() == 0 &&
             cell->level() > this->refine_lower_level_limit && distance < -8))
          {
            needs_refinement_or_coarsening = true;
            break;
          }
      }

  unsigned int do_refine =
    Utilities::MPI::max(static_cast<unsigned int>(needs_refinement_or_coarsening),
                        this->triangulation.get_communicator());

  if (!do_refine)
    return false;

  cell = this->dof_handler.begin_active();
  for (; cell != endc; ++cell, ++ns_cell)
    if (cell->is_locally_owned()) // Cell belongs to the local core or not
      {
        cell->clear_coarsen_flag();
        cell->clear_refine_flag();
        fe_values.reinit(cell);
        vel_values.reinit(ns_cell);
        for (unsigned int d = 0; d < dim; ++d)
          fe_values.get_function_values(this->normal_vector_field.block(d),
                                        ls_gradients[d]);
        fe_values.get_function_values(this->solution.block(0), ls_values);
        vel_values[velocity].get_function_values(this->navier_stokes.solution.block(0),
                                                 u_values);
        double         distance = 0;
        Tensor<1, dim> ls_gradient;
        for (unsigned int q = 0; q < quadrature.size(); ++q)
          {
            for (unsigned int d = 0; d < dim; ++d)
              ls_gradient[d] = ls_gradients[d][q];
            distance = std::max(distance, ls_gradient.norm());
          }

        distance = std::log(distance * this->epsilon_used);

        // try to look ahead and bias the error towards the flow direction
        const double direction = 4. * this->time_stepping.step_size() *
                                 (ls_gradient * u_values[0]) / ls_gradient.norm() /
                                 this->epsilon_used;
        const double mod_distance = distance + direction * ls_values[0];

        bool refine_cell =
          ((cell->level() < upper_level_limit) && (mod_distance > -7 || distance > -4));

        if (refine_cell == true)
          cell->set_refine_flag();
        else if ((cell->level() > this->refine_lower_level_limit) &&
                 (mod_distance < -8 || distance < -5))
          cell->set_coarsen_flag();
      }
  last_refine_step = this->time_stepping.step_no();
  return true;
}



template <int dim>
void
adaflo::LevelSetBaseAlgorithm<dim>::compute_density_on_faces()
{
  if (this->parameters.augmented_taylor_hood == false ||
      this->parameters.density_diff == 0 ||
      this->parameters.linearization == FlowParameters::projection)
    return;

  FEValues<dim>       fe_values(this->mapping,
                          *this->fe,
                          this->face_center_quadrature,
                          update_values);
  std::vector<double> heaviside_values(fe_values.n_quadrature_points);
  AssertDimension(heaviside_values.size(), GeometryInfo<dim>::faces_per_cell);

  for (typename DoFHandler<dim>::active_cell_iterator cell =
         this->dof_handler.begin_active();
       cell != this->dof_handler.end();
       ++cell)
    if (cell->is_locally_owned())
      {
        fe_values.reinit(cell);
        fe_values.get_function_values(heaviside, heaviside_values);
        for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
          this->navier_stokes.set_face_average_density(cell,
                                                       f,
                                                       this->parameters.density +
                                                         heaviside_values[f] *
                                                           this->parameters.density_diff);
      }
}



template <int dim>
void
adaflo::LevelSetBaseAlgorithm<dim>::output_solution(
  const std::string  output_name,
  const unsigned int n_subdivisions) const
{
  if (this->time_stepping.at_tick(this->parameters.output_frequency) == false)
    return;

  if (this->parameters.print_solution_fields == false)
    return;

  TimerOutput::Scope timer(*this->timer, "Create output");

  if (this->time_stepping.step_no() == 0)
    {
      const_cast<LevelSetBaseAlgorithm<dim> *>(this)->compute_curvature();
      const_cast<LevelSetBaseAlgorithm<dim> *>(this)->compute_heaviside();
    }
  const FESystem<dim> joint_fe(this->navier_stokes.get_fe_u(),
                               1,
                               this->navier_stokes.get_fe_p(),
                               1,
                               *this->fe,
                               4,
                               *this->fe,
                               dim);
  DoFHandler<dim>     joint_dof_handler(this->triangulation);
  joint_dof_handler.distribute_dofs(joint_fe);
  Assert(joint_dof_handler.n_dofs() ==
           this->navier_stokes.get_dof_handler_u().n_dofs() +
             this->navier_stokes.get_dof_handler_p().n_dofs() +
             (4 + dim) * this->dof_handler.n_dofs(),
         ExcInternalError());

  LinearAlgebra::distributed::Vector<double> joint_solution;
  {
    IndexSet locally_relevant_joint_dofs(joint_dof_handler.n_dofs());
    DoFTools::extract_locally_relevant_dofs(joint_dof_handler,
                                            locally_relevant_joint_dofs);

    joint_solution.reinit(joint_dof_handler.locally_owned_dofs(),
                          locally_relevant_joint_dofs,
                          this->triangulation.get_communicator());
  }


  {
    std::vector<types::global_dof_index> local_joint_dof_indices(joint_fe.dofs_per_cell);
    std::vector<types::global_dof_index> local_vel_dof_indices(
      this->navier_stokes.get_fe_u().dofs_per_cell);
    std::vector<types::global_dof_index> local_pre_dof_indices(
      this->navier_stokes.get_fe_p().dofs_per_cell);
    std::vector<types::global_dof_index> local_levelset_dof_indices(
      this->fe->dofs_per_cell);

    typename DoFHandler<dim>::active_cell_iterator

      joint_cell    = joint_dof_handler.begin_active(),
      joint_endc    = joint_dof_handler.end(),
      vel_cell      = this->navier_stokes.get_dof_handler_u().begin_active(),
      pre_cell      = this->navier_stokes.get_dof_handler_p().begin_active(),
      levelset_cell = this->dof_handler.begin_active();

    for (; joint_cell != joint_endc;
         ++joint_cell, ++vel_cell, ++pre_cell, ++levelset_cell)
      // Cell belongs to the local core or not

      if (vel_cell->is_locally_owned()) // joint_cell->is_locally_owned()
        {
          joint_cell->get_dof_indices(local_joint_dof_indices);
          vel_cell->get_dof_indices(local_vel_dof_indices);
          pre_cell->get_dof_indices(local_pre_dof_indices);
          levelset_cell->get_dof_indices(local_levelset_dof_indices);

          for (unsigned int i = 0; i < joint_fe.dofs_per_cell; ++i)
            {
              if (joint_dof_handler.locally_owned_dofs().is_element(
                    local_joint_dof_indices[i]) == false)
                continue;

              const unsigned int index = joint_fe.system_to_base_index(i).second;

              if (joint_fe.system_to_base_index(i).first.first == 0)
                {
                  if (joint_fe.system_to_base_index(i).first.second == 0)
                    joint_solution(local_joint_dof_indices[i]) =
#if true
                      this->navier_stokes.solution.block(0)(local_vel_dof_indices[index]);
#else
                      this->navier_stokes.user_rhs.block(0)(local_vel_dof_indices[index]);
#endif
                }
              else if (joint_fe.system_to_base_index(i).first.first == 1)
                {
                  Assert(index < local_pre_dof_indices.size(), ExcInternalError());
                  joint_solution(local_joint_dof_indices[i]) =
                    this->navier_stokes.solution.block(1)(local_pre_dof_indices[index]);
                }
              else if (joint_fe.system_to_base_index(i).first.first == 2)
                {
                  Assert(index < 3 * local_levelset_dof_indices.size(),
                         ExcInternalError());
                  unsigned int cur_index = index % local_levelset_dof_indices.size();
                  if (joint_fe.system_to_base_index(i).first.second == 0)
                    joint_solution(local_joint_dof_indices[i]) =
                      heaviside(local_levelset_dof_indices[cur_index]);
                  else if (joint_fe.system_to_base_index(i).first.second == 1)
                    {
                      joint_solution(local_joint_dof_indices[i]) =
                        this->solution.block(0)(local_levelset_dof_indices[cur_index]);
                    }
                  else if (joint_fe.system_to_base_index(i).first.second == 2)
                    {
                      joint_solution(local_joint_dof_indices[i]) =
                        this->solution.block(1)(local_levelset_dof_indices[cur_index]);
                    }
                  else if (joint_fe.system_to_base_index(i).first.second == 3)
                    {
                      joint_solution(local_joint_dof_indices[i]) =
                        vel_cell->subdomain_id();
                    }
                }
              else if (joint_fe.system_to_base_index(i).first.first == 3)
                {
                  Assert(index < dim * local_levelset_dof_indices.size(),
                         ExcInternalError());
                  unsigned int cur_index = index % local_levelset_dof_indices.size();
                  joint_solution(local_joint_dof_indices[i]) =
                    this->normal_vector_field.block(
                      joint_fe.system_to_base_index(i).first.second)(
                      local_levelset_dof_indices[cur_index]);
                }
            }
        }
  }
  joint_solution.update_ghost_values();

#if true
  std::vector<std::string> joint_solution_names(dim, "velocity");
#else
  std::vector<std::string> joint_solution_names(dim, "user_rhs");
#endif
  joint_solution_names.push_back("pressure");
  joint_solution_names.push_back("heaviside");
  joint_solution_names.push_back("level_set");
  joint_solution_names.push_back(this->curvature_name);
  joint_solution_names.push_back("owner");
  for (unsigned int d = 0; d < dim; ++d)
    joint_solution_names.push_back("normal_vector");


  Vector<double> refine_usual(this->dof_handler.get_triangulation().n_active_cells());
  Vector<double> refine_bias(this->dof_handler.get_triangulation().n_active_cells());
  Vector<double> direction_val(this->dof_handler.get_triangulation().n_active_cells());

  std::vector<Point<1>> point(2);
  point[0][0] = 0.05;
  point[1][0] = 0.95;
  Quadrature<1>   quadrature_1d(point);
  Quadrature<dim> quadrature(quadrature_1d);
  FEValues<dim>   fe_values(*this->fe, quadrature, update_values);
  FEValues<dim>   vel_values(this->navier_stokes.get_fe_u(), quadrature, update_values);
  std::vector<std::vector<double>> ls_gradients(dim,
                                                std::vector<double>(quadrature.size()));
  std::vector<double>              ls_values(quadrature.size());
  std::vector<Tensor<1, dim>>      u_values(quadrature.size());

  const FEValuesExtractors::Vector               velocity(0);
  typename DoFHandler<dim>::active_cell_iterator cell = this->dof_handler.begin_active(),
                                                 ns_cell =
                                                   this->navier_stokes.get_dof_handler_u()
                                                     .begin_active(),
                                                 endc = this->dof_handler.end();
  for (unsigned int c = 0; cell != endc; ++cell, ++ns_cell, ++c)
    if (cell->is_locally_owned()) // Cell belongs to the local core or not
      {
        cell->clear_coarsen_flag();
        cell->clear_refine_flag();
        fe_values.reinit(cell);
        vel_values.reinit(ns_cell);
        for (unsigned int d = 0; d < dim; ++d)
          fe_values.get_function_values(this->normal_vector_field.block(d),
                                        ls_gradients[d]);
        fe_values.get_function_values(this->solution.block(0), ls_values);
        vel_values[velocity].get_function_values(this->navier_stokes.solution.block(0),
                                                 u_values);
        double         distance = 0;
        Tensor<1, dim> ls_gradient;
        for (unsigned int q = 0; q < quadrature.size(); ++q)
          {
            for (unsigned int d = 0; d < dim; ++d)
              ls_gradient[d] = ls_gradients[d][q];
            distance = std::max(distance, ls_gradient.norm());
          }

        distance = std::log(distance * this->epsilon_used);

        // try to look ahead and bias the error towards the flow direction
        const double direction = 20. * this->time_stepping.step_size() *
                                 (ls_gradient * u_values[0]) / ls_gradient.norm() /
                                 this->epsilon_used;
        const double mod_distance = distance + direction * ls_values[0];

        refine_usual(c)  = distance > -7;
        refine_bias(c)   = mod_distance > -7 || distance > -4;
        direction_val(c) = direction;
      }

  DataOut<dim> data_out;

  data_out.attach_dof_handler(joint_dof_handler);


  std::vector<DataComponentInterpretation::DataComponentInterpretation>
    data_component_interpretation(dim + 5 + dim,
                                  DataComponentInterpretation::component_is_scalar);
  for (unsigned int i = 0; i < dim; ++i)
    data_component_interpretation[i] =
      DataComponentInterpretation::component_is_part_of_vector;
  for (unsigned int i = dim + 5; i < 2 * dim + 5; ++i)
    data_component_interpretation[i] =
      DataComponentInterpretation::component_is_part_of_vector;


  data_out.add_data_vector(joint_solution,
                           joint_solution_names,
                           DataOut<dim>::type_dof_data,
                           data_component_interpretation);
  data_out.add_data_vector(refine_usual, "Standard_refine");
  data_out.add_data_vector(refine_bias, "Biased_refine");
  data_out.add_data_vector(direction_val, "Biased_refine_direction");

  const unsigned int n_patches = n_subdivisions == 0 ?
                                   std::min(this->parameters.velocity_degree,
                                            this->parameters.concentration_subdivisions) :
                                   n_subdivisions;
  data_out.build_patches(this->mapping, n_patches);

  this->write_data_output(output_name,
                          this->time_stepping,
                          this->parameters.output_frequency,
                          this->triangulation,
                          data_out);
}



template class adaflo::LevelSetBaseAlgorithm<2>;
template class adaflo::LevelSetBaseAlgorithm<3>;
