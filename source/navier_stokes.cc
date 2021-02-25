// --------------------------------------------------------------------------
//
// Copyright (C) 2009 - 2016 by the adaflo authors
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
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/utilities.h>

#include <deal.II/distributed/grid_refinement.h>

#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_q_dg0.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_fe.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>

#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/vector_tools.h>

#include <adaflo/navier_stokes.h>
#include <adaflo/util.h>

#include <fstream>
#include <iomanip>
#include <iostream>



template <int dim>
NavierStokes<dim>::NavierStokes(
  const FlowParameters &                            parameters,
  Triangulation<dim> &                              triangulation_in,
  TimerOutput *                                     external_timer,
  std::shared_ptr<helpers::BoundaryDescriptor<dim>> boundary_descriptor)
  : NavierStokes(parameters.use_simplex_mesh ?
                   static_cast<const Mapping<dim> &>(
                     MappingFE<dim>(FE_SimplexP<dim>(1))) :
                   static_cast<const Mapping<dim> &>(MappingQ<dim>(3)),
                 parameters,
                 triangulation_in,
                 external_timer,
                 boundary_descriptor)
{}


template <int dim>
NavierStokes<dim>::NavierStokes(
  const Mapping<dim> &                              mapping,
  const FlowParameters &                            parameters,
  Triangulation<dim> &                              triangulation_in,
  TimerOutput *                                     external_timer,
  std::shared_ptr<helpers::BoundaryDescriptor<dim>> boundary_descriptor)
  : FlowBaseAlgorithm<dim>(std::shared_ptr<Mapping<dim>>(mapping.clone()))
  , user_rhs(2)
  , solution(2)
  , solution_old(2)
  , solution_old_old(2)
  , solution_update(2)
  , time_stepping(parameters)
  , parameters(parameters)
  , n_mpi_processes(Utilities::MPI::n_mpi_processes(get_communicator(triangulation_in)))
  , this_mpi_process(Utilities::MPI::this_mpi_process(get_communicator(triangulation_in)))
  , pcout(std::cout, this_mpi_process == 0)
  , triangulation(triangulation_in)
  , fe_u(parameters.use_simplex_mesh ?
           static_cast<const FiniteElement<dim> &>(
             FE_SimplexP<dim>(parameters.velocity_degree)) :
           static_cast<const FiniteElement<dim> &>(
             FE_Q<dim>(QGaussLobatto<1>(parameters.velocity_degree + 1))),
         dim)
  , fe_p(parameters.use_simplex_mesh ?
           static_cast<const FiniteElement<dim> &>(
             FE_SimplexP<dim>(parameters.velocity_degree - 1)) :
           (parameters.augmented_taylor_hood ?
              static_cast<const FiniteElement<dim> &>(
                FE_Q_DG0<dim>(parameters.velocity_degree - 1)) :
              static_cast<const FiniteElement<dim> &>(
                FE_Q<dim>(QGaussLobatto<1>(parameters.velocity_degree)))),
         1)
  , dof_handler_u(triangulation)
  , dof_handler_p(triangulation)
  , navier_stokes_matrix(parameters,
                         dof_index_u,
                         dof_index_p,
                         quad_index_u,
                         quad_index_p,
                         solution_old,
                         solution_old_old)
  , system_rhs(2)
  , const_rhs(2)
  , preconditioner(parameters, *this, triangulation, constraints_u)
  , dofs_distributed(false)
  , system_is_setup(false)
  , n_iterations_last_prec_update(0)
  , time_step_last_prec_update(0)
  , update_preconditioner(true)
  , update_preconditioner_frequency(0)
{
  if (boundary_descriptor.get() != 0)
    this->boundary = boundary_descriptor;

  // if we own the timer (not obtained from another class), reset
  // it. otherwise, just set the pointer

  if (external_timer == 0)
    {
      const auto output_frequency =
        parameters.output_wall_times ? TimerOutput::summary : TimerOutput::never;
      timer =
        std::make_shared<TimerOutput>(pcout, output_frequency, TimerOutput::wall_times);
    }
  else
    timer.reset(external_timer, helpers::DummyDeleter<TimerOutput>());
}



template <int dim>
NavierStokes<dim>::~NavierStokes()
{
  solver_memory.release_unused_memory();
  GrowingVectorMemory<
    LinearAlgebra::distributed::Vector<double>>::release_unused_memory();
  GrowingVectorMemory<
    LinearAlgebra::distributed::BlockVector<double>>::release_unused_memory();
}


template <int dim>
std::pair<unsigned int, unsigned int>
NavierStokes<dim>::n_dofs() const
{
  Assert(dofs_distributed == true, ExcInternalError());
  return std::pair<unsigned int, unsigned int>(dof_handler_u.n_dofs(),
                                               dof_handler_p.n_dofs());
}



template <int dim>
void
NavierStokes<dim>::print_n_dofs() const
{
  std::pair<unsigned int, unsigned int> n_dofs = this->n_dofs();
  const double min_cell_diameter = -Utilities::MPI::max(-triangulation.last()->diameter(),
                                                        get_communicator(triangulation));

  pcout << " Number of active cells: " << triangulation.n_global_active_cells() << "."
        << std::endl

        << " Number of degrees of freedom (velocity/pressure): "
        << n_dofs.first + n_dofs.second << " (" << n_dofs.first << " + " << n_dofs.second
        << ")." << std::endl

        << " Approximate size last cell: " << min_cell_diameter / std::sqrt(dim)
        << std::endl;
}



template <int dim>
void
NavierStokes<dim>::distribute_dofs()
{
  timer->enter_subsection("NS distribute DoFs.");

  solver_memory.release_unused_memory();
  navier_stokes_matrix.clear();
  hanging_node_constraints_u.clear();
  hanging_node_constraints_p.clear();
  constraints_u.clear();
  constraints_p.clear();
  matrix_free.reset();
  preconditioner.clear();

  dof_handler_u.distribute_dofs(fe_u);
  dof_handler_p.distribute_dofs(fe_p);

  if (parameters.precondition_velocity == FlowParameters::u_ilu)
    DoFRenumbering::Cuthill_McKee(dof_handler_u, false, false);

  IndexSet relevant_dofs_p, relevant_dofs_u;
  DoFTools::extract_locally_relevant_dofs(dof_handler_p, relevant_dofs_p);
  hanging_node_constraints_p.reinit(relevant_dofs_p);
  constraints_p.reinit(relevant_dofs_p);
  DoFTools::extract_locally_relevant_dofs(dof_handler_u, relevant_dofs_u);
  hanging_node_constraints_u.reinit(relevant_dofs_u);
  constraints_u.reinit(relevant_dofs_u);

  dofs_distributed = true;
  system_is_setup  = false;

  timer->leave_subsection();
}



template <int dim>
void
NavierStokes<dim>::initialize_data_structures()
{
  if (system_is_setup == true)
    return;

  timer->enter_subsection("NS setup matrix and vectors.");
  Assert(dofs_distributed == true, ExcInternalError());

  // Now, the constraint matrix for hanging nodes and boundary conditions. We
  // start with the hanging nodes for both components and the periodicity
  // constraints which can be set the same way as hanging node constraints.
  DoFTools::make_hanging_node_constraints(dof_handler_u, hanging_node_constraints_u);
  DoFTools::make_hanging_node_constraints(dof_handler_p, hanging_node_constraints_p);
  for (unsigned int d = 0; d < dim; ++d)
    if (this->boundary->periodic_boundaries[d] !=
        std::pair<types::boundary_id, types::boundary_id>(-1, -1))
      {
        const types::boundary_id in  = this->boundary->periodic_boundaries[d].first;
        const types::boundary_id out = this->boundary->periodic_boundaries[d].second;
        AssertThrow(
          this->boundary->open_conditions_p.find(in) ==
              this->boundary->open_conditions_p.end() &&
            this->boundary->open_conditions_p.find(out) ==
              this->boundary->open_conditions_p.end() &&
            this->boundary->dirichlet_conditions_u.find(in) ==
              this->boundary->dirichlet_conditions_u.end() &&
            this->boundary->dirichlet_conditions_u.find(out) ==
              this->boundary->dirichlet_conditions_u.end() &&
            this->boundary->no_slip.find(in) == this->boundary->no_slip.end() &&
            this->boundary->no_slip.find(out) == this->boundary->no_slip.end() &&
            this->boundary->symmetry.find(in) == this->boundary->symmetry.end() &&
            this->boundary->symmetry.find(out) == this->boundary->symmetry.end(),
          ExcMessage("Cannot mix periodic boundary conditions with "
                     "other types of boundary conditions on same "
                     "boundary!"));
        AssertThrow(in != out,
                    ExcMessage("The two faces for periodic boundary conditions "
                               "must have different boundary indicators!"));
        DoFTools::make_periodicity_constraints(
          dof_handler_u, in, out, d, hanging_node_constraints_u);
        DoFTools::make_periodicity_constraints(
          dof_handler_p, in, out, d, hanging_node_constraints_p);
      }

  for (typename std::set<types::boundary_id>::const_iterator it =
         this->boundary->symmetry.begin();
       it != this->boundary->symmetry.end();
       ++it)
    AssertThrow(this->boundary->open_conditions_p.find(*it) ==
                    this->boundary->open_conditions_p.end() &&
                  this->boundary->no_slip.find(*it) == this->boundary->no_slip.end() &&
                  this->boundary->dirichlet_conditions_u.find(*it) ==
                    this->boundary->dirichlet_conditions_u.end(),
                ExcMessage("Cannot mix symmetry boundary conditions with "
                           "other boundary conditions on same boundary!"));


  VectorTools::compute_no_normal_flux_constraints(dof_handler_u,
                                                  0,
                                                  this->boundary->symmetry,
                                                  hanging_node_constraints_u,
                                                  this->mapping);
  VectorTools::compute_normal_flux_constraints(dof_handler_u,
                                               0,
                                               this->boundary->normal_flux,
                                               hanging_node_constraints_u,
                                               this->mapping);

  // Now generate the rest of the constraints for the velocity
  constraints_u.merge(hanging_node_constraints_u);
  {
    Functions::ZeroFunction<dim>                        zero_func(dim);
    std::map<types::boundary_id, const Function<dim> *> homogeneous_dirichlet;
    for (typename std::map<types::boundary_id,
                           std::shared_ptr<Function<dim>>>::const_iterator it =
           this->boundary->dirichlet_conditions_u.begin();
         it != this->boundary->dirichlet_conditions_u.end();
         ++it)
      {
        AssertThrow(this->boundary->open_conditions_p.find(it->first) ==
                      this->boundary->open_conditions_p.end(),
                    ExcMessage("Cannot mix velocity Dirichlet conditions with "
                               "open/pressure boundary conditions on same "
                               "boundary!"));
        homogeneous_dirichlet[it->first] = &zero_func;
      }

    // no-slip boundaries
    for (typename std::set<types::boundary_id>::const_iterator it =
           this->boundary->no_slip.begin();
         it != this->boundary->no_slip.end();
         ++it)
      {
        AssertThrow(this->boundary->open_conditions_p.find(*it) ==
                      this->boundary->open_conditions_p.end(),
                    ExcMessage("Cannot mix velocity Dirichlet conditions with "
                               "open/pressure boundary conditions on same "
                               "boundary!"));
        homogeneous_dirichlet[*it] = &zero_func;
      }

    VectorTools::interpolate_boundary_values(this->mapping,
                                             dof_handler_u,
                                             homogeneous_dirichlet,
                                             constraints_u);
  }

  hanging_node_constraints_u.close();
  constraints_u.close();
  AssertThrow(constraints_u.has_inhomogeneities() == false,
              ExcMessage("Constraint matrix for u has inhomogeneities which "
                         "is not allowed."));

  // Next set the hanging node constraints and boundary conditions for
  // pressure
  constraints_p.merge(hanging_node_constraints_p,
                      AffineConstraints<double>::right_object_wins);
  hanging_node_constraints_p.close();
  constraints_p.close();
  AssertThrow(constraints_p.has_inhomogeneities() == false,
              ExcMessage("Constraint matrix for p has inhomogeneities which "
                         "is not allowed."));

  preconditioner.initialize_matrices(dof_handler_u, dof_handler_p, constraints_p);

  system_is_setup       = true;
  update_preconditioner = true;
  timer->leave_subsection();
}



template <int dim>
void
NavierStokes<dim>::setup_problem(const Function<dim> &initial_velocity_field,
                                 const Function<dim> &)
{
  if (parameters.use_simplex_mesh)
    AssertDimension(parameters.global_refinements, 0);

  // if we should to more than 15 refinements, this can't be right: We would
  // get 1e9 as many elements in 2d and 3e13 in 3d! The user likely used this
  // variables for specifying how often to refine a rectangle...
  if (parameters.global_refinements < 15)
    triangulation.refine_global(parameters.global_refinements);

  distribute_dofs();
  initialize_data_structures();
  initialize_matrix_free();

  if (!time_stepping.at_end())
    {
      VectorTools::interpolate(this->mapping,
                               get_dof_handler_u(),
                               initial_velocity_field,
                               solution.block(0));
      hanging_node_constraints_u.distribute(solution.block(0));
    }
  solution.update_ghost_values();
  solution_old.update_ghost_values();
}



template <int dim>
void
NavierStokes<dim>::initialize_matrix_free(MatrixFree<dim> *  external_matrix_free,
                                          const unsigned int dof_index_u,
                                          const unsigned int dof_index_p,
                                          const unsigned int quad_index_u,
                                          const unsigned int quad_index_p)
{
  this->dof_index_u  = dof_index_u;
  this->dof_index_p  = dof_index_p;
  this->quad_index_u = quad_index_u;
  this->quad_index_p = quad_index_p;

  if (external_matrix_free != 0)
    {
      matrix_free.reset(const_cast<MatrixFree<dim> *>(external_matrix_free),
                        helpers::DummyDeleter<MatrixFree<dim>>());
    }
  else
    {
      matrix_free.reset(new MatrixFree<dim>(),
                        helpers::DummyDeleter<MatrixFree<dim>>(true));
      typename MatrixFree<dim>::AdditionalData data;

      // writing into an Epetra_FECrsMatrix is not thread-safe (non-local
      // data), so do not allow parallelism in case we use more than one
      // processor
      data.mapping_update_flags = data.mapping_update_flags | update_quadrature_points;
      data.tasks_parallel_scheme =
        Utilities::MPI::n_mpi_processes(get_communicator(triangulation)) > 1 ?
          MatrixFree<dim>::AdditionalData::none :
          MatrixFree<dim>::AdditionalData::partition_color;
      if (parameters.velocity_degree == 2)
        data.tasks_block_size = 16;
      else
        data.tasks_block_size = 2;
      data.store_plain_indices = true;
      std::vector<const DoFHandler<dim> *> dof_handlers(2);
      dof_handlers[dof_index_u] = &dof_handler_u;
      dof_handlers[dof_index_p] = &dof_handler_p;
      std::vector<const AffineConstraints<double> *> constraints(2);
      constraints[dof_index_u] = &constraints_u;
      constraints[dof_index_p] = &constraints_p;
      std::vector<Quadrature<dim>> quadratures(2);
      if (parameters.use_simplex_mesh)
        {
          quadratures[quad_index_u] = QGaussSimplex<dim>(parameters.velocity_degree + 1);
          quadratures[quad_index_p] = QGaussSimplex<dim>(parameters.velocity_degree);
        }
      else
        {
          quadratures[quad_index_u] = QGauss<dim>(parameters.velocity_degree + 1);
          quadratures[quad_index_p] = QGauss<dim>(parameters.velocity_degree);
        }
      matrix_free->reinit(this->mapping, dof_handlers, constraints, quadratures, data);
    }
  navier_stokes_matrix.initialize(*matrix_free,
                                  time_stepping,
                                  !this->boundary->pressure_fix.empty());
  preconditioner.set_system_matrix(navier_stokes_matrix);

  solution_update.reinit(2);
  matrix_free->initialize_dof_vector(solution_update.block(0), dof_index_u);
  matrix_free->initialize_dof_vector(solution_update.block(1), dof_index_p);
  solution_update.collect_sizes();

  solution.reinit(solution_update);
  solution_old.reinit(solution_update);
  solution_old_old.reinit(solution_update);
  system_rhs.reinit(solution_update);
  const_rhs.reinit(solution_update);
  user_rhs.reinit(solution_update);

  // if applicable, finish the transfer of the solution
  if (sol_trans_u.get() != 0)
    {
      Assert(sol_trans_p.get() != 0, ExcInternalError());

      std::vector<LinearAlgebra::distributed::Vector<double> *> new_grid_solutions(2);

      new_grid_solutions[0] = &solution.block(0);
      new_grid_solutions[1] = &solution_old.block(0);
      sol_trans_u->interpolate(new_grid_solutions);
      sol_trans_u.reset();
      hanging_node_constraints_u.distribute(solution.block(0));
      hanging_node_constraints_u.distribute(solution_old.block(0));

      new_grid_solutions[0] = &solution.block(1);
      new_grid_solutions[1] = &solution_old.block(1);
      if (parameters.linearization == FlowParameters::projection)
        new_grid_solutions.push_back(&solution_old_old.block(1));
      sol_trans_p->interpolate(new_grid_solutions);
      sol_trans_p.reset();

      hanging_node_constraints_p.distribute(solution.block(1));
      solution.update_ghost_values();

      hanging_node_constraints_p.distribute(solution_old.block(1));
      solution_old.update_ghost_values();

      if (parameters.linearization == FlowParameters::projection)
        {
          hanging_node_constraints_p.distribute(solution_old_old.block(1));
          solution_old_old.block(1).update_ghost_values();
        }
    }
}



template <int dim>
void
NavierStokes<dim>::assemble_preconditioner()
{
  timer->enter_subsection("NS assemble preconditioner.");

  // release solver memory since we need a lot of memory for assembly
  solver_memory.release_unused_memory();
  preconditioner.assemble_matrices();

  timer->leave_subsection();
}



template <int dim>
void
NavierStokes<dim>::build_preconditioner()
{
  assemble_preconditioner();

  // if time step is large then ILU is not a good preconditioner for the
  // velocity-velocity matrix (when the mass matrix is sufficiently large to
  // dominate the matrix)
  timer->enter_subsection("NS build preconditioner.");

  if (parameters.output_verbosity > 0)
    {
      std::ostringstream update_mode;
      if (parameters.output_verbosity == 1)
        update_mode << "/";
      else if (parameters.output_verbosity >= 2)
        update_mode << "    ";
      if (parameters.precondition_velocity == FlowParameters::u_ilu)
        update_mode << (parameters.output_verbosity >= 2 ? "ILU " : "ILU");
      else if (parameters.precondition_velocity == FlowParameters::u_ilu_scalar)
        update_mode << "ILUs";
      else if (parameters.precondition_velocity == FlowParameters::u_amg)
        update_mode << (parameters.output_verbosity >= 2 ? "AMG " : "AMG");
      else if (parameters.precondition_velocity == FlowParameters::u_amg_linear)
        update_mode << "AMGl";
      if (parameters.output_verbosity >= 2)
        update_mode << "   ";
      pcout << update_mode.str();
    }

  preconditioner.compute();

  timer->leave_subsection();
}



template <int dim>
std::pair<unsigned int, double>
NavierStokes<dim>::solve_system(const double linear_tolerance)
{
  if (parameters.linearization == FlowParameters::projection)
    return preconditioner.solve_projection_system(
      solution, solution_update, system_rhs, solution_old.block(1), *timer);

  timer->enter_subsection("NS solve system.");
  Timer time;

  solution_update = 0;
  SolverControl solver_control_simple(std::min(parameters.iterations_before_inner_solvers,
                                               parameters.max_lin_iteration),
                                      linear_tolerance);

  SolverControl solver_control_strong(
    std::max<int>((int)parameters.max_lin_iteration -
                    (int)parameters.iterations_before_inner_solvers,
                  0),
    linear_tolerance);

  // first try cheap solver that does not make inner iterations, if it does
  // not succeed, throw the more powerful (but more expensive) solver at
  // it. Note that we use FGMRES for that case as there are inner iterations
  // which make the preconditioner non-linear
  double residual = 1.;
  try
    {
      preconditioner.do_inner_solves = false;
      if (preconditioner.is_variable())
        {
          SolverFGMRES<LinearAlgebra::distributed::BlockVector<double>> solver(
            solver_control_simple,
            solver_memory,
            SolverFGMRES<LinearAlgebra::distributed::BlockVector<double>>::AdditionalData(
              50));

          solver.solve(navier_stokes_matrix, solution_update, system_rhs, preconditioner);
        }
      else
        {
          SolverGMRES<LinearAlgebra::distributed::BlockVector<double>> solver(
            solver_control_simple,
            solver_memory,
            SolverGMRES<LinearAlgebra::distributed::BlockVector<double>>::AdditionalData(
              50, true));

          solver.solve(navier_stokes_matrix, solution_update, system_rhs, preconditioner);
        }
      residual = solver_control_simple.last_value();
    }
  catch (const SolverControl::NoConvergence &)
    {
      if (parameters.iterations_before_inner_solvers < parameters.max_lin_iteration)
        {
          preconditioner.do_inner_solves = true;
          SolverFGMRES<LinearAlgebra::distributed::BlockVector<double>> solver(
            solver_control_strong,
            solver_memory,
            SolverFGMRES<LinearAlgebra::distributed::BlockVector<double>>::AdditionalData(
              50));

          // if also the expensive solver fails, still catch the assertion
          // since we do not want to fail because of the linear solver
          try
            {
              solver.solve(navier_stokes_matrix,
                           solution_update,
                           system_rhs,
                           preconditioner);
            }
          catch (const SolverControl::NoConvergence &)
            {}

          residual = solver_control_strong.last_value();
        }
      else
        residual = solver_control_simple.last_value();
    }

  constraints_u.distribute(solution_update.block(0));
  constraints_p.distribute(solution_update.block(1));

  solver_timers[0].second += time.wall_time();
  solver_timers[0].first++;

  timer->leave_subsection();

  return std::pair<unsigned int, double>(solver_control_simple.last_step() +
                                           solver_control_strong.last_step(),
                                         residual);
}



template <int dim>
void
NavierStokes<dim>::init_time_advance(const bool print_time_info)
{
  Assert(system_is_setup == true, ExcMessage("System has not yet been set up!"));

  time_stepping.next();

  // Calculate extrapolated solution (= initial guess for nonlinear iteration)
  // at time n+1. We always do this by using two old solutions. The quality of
  // the initial guess for the nonlinear iteration is considerably improved
  // from this extrapolation than a naive use of the old solution value. Use
  // solution_update for a temporary vector to save this information to.
  const unsigned int n_blocks =
    parameters.linearization == FlowParameters::projection ? 1 : 2;
  for (unsigned int block = 0; block < n_blocks; ++block)
    {
      const unsigned int n       = solution.block(block).local_size();
      double *           cur     = solution.block(block).begin();
      double *           old     = solution_old.block(block).begin();
      double *           old_old = solution_old_old.block(block).begin();

      for (unsigned int i = 0; i < n; ++i)
        {
          const double tmp = time_stepping.extrapolate(cur[i], old[i]);
          old_old[i]       = old[i];
          old[i]           = cur[i];
          cur[i]           = tmp;
        }
    }

  // For projection, compute p^{n+1,*} = p^n + 4/3 phi^n - 1/3 phi^{n-1} as an
  // extrapolation of old pressure values to the new time as a means to avoid
  // updating the velocity. Also keep p^n temporarily in solution_update while
  // computing the residual. We swap it with the solution before overwriting
  // solution_update
  if (parameters.linearization == FlowParameters::projection)
    {
      if (time_stepping.step_no() > 1)
        {
          const unsigned int n       = solution.block(1).local_size();
          double *           cur     = solution.block(1).begin();
          double *           upd     = solution_update.block(1).begin();
          double *           old     = solution_old.block(1).begin();
          double *           old_old = solution_old_old.block(1).begin();
          for (unsigned int i = 0; i < n; ++i)
            {
              const double tmp =
                cur[i] - time_stepping.weight_old() / time_stepping.weight() * old[i] -
                time_stepping.weight_old_old() / time_stepping.weight() * old_old[i];
              old_old[i] = old[i];
              upd[i]     = cur[i];
              cur[i]     = tmp;
            }
        }
      // else if (time_stepping.step_no() == 2)
      // solution_old_old.block(1) = solution_old.block(1);
      else if (time_stepping.step_no() == 1)
        {
          solution_old.block(1)     = 0.;
          solution_old_old.block(1) = 0.;
        }
    }

  // if (time_stepping.weight_has_changed () == true)
  // update_preconditioner = true;

  if (print_time_info)
    {
      const unsigned int time_step_number = time_stepping.step_no();
      {
        const double time      = time_stepping.now();
        const double old_time  = time_stepping.previous();
        const double step_size = time_stepping.step_size();

        pcout << std::endl << "Time step #" << time_step_number << ", ";
        std::cout.precision(3);
        pcout << "advancing from t_n-1 = " << old_time << " to t = " << time << " "
              << "(dt = " << step_size;
        pcout << "). " << std::endl;
      }
    }

  timer->enter_subsection("NS apply boundary conditions.");
  // This also calls update_ghost_values on all vectors
  apply_boundary_conditions();

  timer->leave_subsection();
}



template <int dim>
unsigned int
NavierStokes<dim>::advance_time_step()
{
  init_time_advance();
  return evaluate_time_step();
}



template <int dim>
unsigned int
NavierStokes<dim>::evaluate_time_step()
{
  const double initial_residual = compute_initial_residual(true);
  return solve_nonlinear_system(initial_residual);
}



template <int dim>
double
NavierStokes<dim>::compute_residual()
{
  TimerOutput::Scope scope(*timer, "NS assemble nonlinear residual.");
  system_rhs.equ(1., const_rhs);
  navier_stokes_matrix.residual(system_rhs, solution, user_rhs);
  navier_stokes_matrix.apply_pressure_average_projection(system_rhs.block(1));

  const double res_u = system_rhs.block(0).l2_norm();
  const double res_p = system_rhs.block(1).l2_norm();
  const double res   = std::sqrt(res_u * res_u + res_p * res_p);

  if (parameters.output_verbosity == 1)
    pcout << "[" << res;
  else if (parameters.output_verbosity == 2 && this_mpi_process == 0)
    std::printf("   %-12.3e ", res);
  else if (parameters.output_verbosity == 3 && this_mpi_process == 0)
    std::printf("   %-11.3e %-12.3e ", res_u, res_p);

  return res;
}



template <int dim>
double
NavierStokes<dim>::compute_initial_residual(const bool)
{
  if (parameters.output_verbosity == 1)
    pcout << "  Residual/iterations: ";
  else if (parameters.output_verbosity == 2)
    {
      pcout << std::endl
            << "   Nonlin Res     Prec Upd     Increment   Lin Iter     Lin Res"
            << std::endl
            << "   ____________________________________________________________"
            << std::endl;
    }
  else if (parameters.output_verbosity == 3)
    {
      pcout
        << std::endl
        << "   NL Resid u  NL Resid p     Prec Upd     Increm u   Increm p   Lin Iter     Lin Res"
        << std::endl
        << "   __________________________________________________________________________________"
        << std::endl;
    }

  return compute_residual();
}



template <int dim>
unsigned int
NavierStokes<dim>::solve_nonlinear_system(const double initial_residual)
{
  Timer        nl_timer;
  unsigned int step = 0;

  // for projection, restore the actual value p^n here
  if (parameters.linearization == FlowParameters::projection)
    solution.block(1).swap(solution_update.block(1));

  std::pair<unsigned int, double> convergence(0, 0.);
  unsigned int                    n_tot_iterations = 0;
  bool                            premature_update = false;

  double res = initial_residual;

  // always do at least one iteration
  for (; step < parameters.max_nl_iteration; ++step)
    {
      // compute linear tolerance and solve
      double linear_tolerance = parameters.tol_lin_iteration;
      if (parameters.rel_lin_iteration)
        {
          // guess that we do not need more than 0.4 the nonlinear tolerance
          // (otherwise, we are converging slowly anyway). if not fully
          // implicit, we have a linear system and should directly try to
          // reach the tolerance
          if (res * parameters.tol_lin_iteration < 0.5 * parameters.tol_nl_iteration ||
              !(parameters.linearization == FlowParameters::coupled_implicit_newton ||
                parameters.linearization == FlowParameters::coupled_implicit_picard))
            linear_tolerance = 0.5 * parameters.tol_nl_iteration;
          else
            linear_tolerance =
              std::min(parameters.tol_lin_iteration * res, parameters.tol_lin_iteration);
        }

      // update preconditioner if in first step of NL iteration and so
      // requested
      if (step == 0 && update_preconditioner)
        build_preconditioner();
      else if ((!premature_update && time_stepping.step_no() > 1 &&
                (n_tot_iterations > 1.5 * n_iterations_last_prec_update)) ||
               (parameters.physical_type == FlowParameters::incompressible_stationary &&
                step % 6 == 1))
        {
          build_preconditioner();
          premature_update = true;
        }
      else if (parameters.output_verbosity >= 2)
        pcout << "    ---    ";

      convergence = solve_system(linear_tolerance);
      solution += solution_update;

      n_tot_iterations += convergence.first;
      if (parameters.output_verbosity == 1)
        pcout << "/" << convergence.first << "] " << std::flush;
      else if (parameters.output_verbosity == 2)
        {
          const double update_norm = solution_update.l2_norm();
          if (this_mpi_process == 0)
            std::printf("    %-5.2e     %4d       %-5.2e\n",
                        update_norm,
                        convergence.first,
                        convergence.second);
        }
      else if (parameters.output_verbosity == 3)
        {
          const double incr_u = solution_update.block(0).l2_norm();
          const double incr_p = solution_update.block(1).l2_norm();
          if (this_mpi_process == 0)
            std::printf("    %-5.2e   %-5.2e    %4d       %-5.2e\n",
                        incr_u,
                        incr_p,
                        convergence.first,
                        convergence.second);
        }

      if (parameters.linearization != FlowParameters::coupled_implicit_newton &&
          parameters.linearization != FlowParameters::coupled_implicit_picard)
        {
          if (parameters.output_verbosity == 1)
            pcout << "[" << convergence.second << "/conv.]" << std::endl;
          else if (parameters.output_verbosity >= 2)
            pcout << std::endl;
          break;
        }

      res = compute_residual();
      // check for convergence
      if (res < parameters.tol_nl_iteration)
        {
          if (parameters.output_verbosity == 1)
            pcout << "/conv.]" << std::endl;
          else if (parameters.output_verbosity >= 2 && this_mpi_process == 0)
            std::printf(" converged.\n\n");
          break;
        }
    }

  // After the iteration, check whether the convergence of the linear solver
  // deteriorated compared to the last time we rebuilt it (i.e., it needs 20%
  // more iterations than the value we achieved after the last rebuilt, or it
  // was only updated once in the beginning of the time stepping): if yes,
  // need to update it. We do not rebuild the preconditioner matrix during the
  // steps of the Newton iteration because we expect only little change there

  if (update_preconditioner_frequency > 0 &&
      time_stepping.step_no() % (50 * update_preconditioner_frequency) == 0)
    update_preconditioner_frequency = 0;

  if (update_preconditioner == true)
    {
      n_iterations_last_prec_update = n_tot_iterations;
      time_step_last_prec_update    = time_stepping.step_no();
      update_preconditioner         = false;
    }

  if (n_tot_iterations > 1.2 * n_iterations_last_prec_update)
    {
      if (premature_update == true ||
          n_tot_iterations > 2 * n_iterations_last_prec_update)
        update_preconditioner_frequency =
          time_stepping.step_no() - time_step_last_prec_update;
      update_preconditioner = true;
    }

  if ((time_step_last_prec_update < 3 && time_stepping.step_no() > 14) ||
      time_stepping.step_no() < 2)
    update_preconditioner = true;

  if (update_preconditioner == false && premature_update == false &&
      update_preconditioner_frequency > 0 &&
      time_stepping.step_no() + 1 - time_step_last_prec_update >=
        update_preconditioner_frequency)
    {
      update_preconditioner = true;
    }

  if (step == parameters.max_nl_iteration)
    {
      if (parameters.output_verbosity == 1)
        pcout << "]" << std::endl;
      pcout << "Warning: nonlinear iteration did not converge!" << std::endl;
    }

  solution.block(0).update_ghost_values();
  solution.block(1).update_ghost_values();

  // finally, if we need to fix the pressure, apply a shift so that the first
  // value on the boundary exactly satisfies the boundary condition
  if (!this->boundary->pressure_fix.empty())
    {
      TimerOutput::Scope(*timer, "NS apply boundary conditions.");

      // Find cell at boundary with the given index. Take the first such cell
      // and compute the distance between the numerical solution and the given
      // analytic solution. Then add a shift to the vector (only the constant
      // modes for the FE_Q part)
      std::vector<types::global_dof_index> dof_indices(fe_p.dofs_per_cell);
      std::pair<long int, double> distance(std::numeric_limits<long int>::max(), 0.);
      QGauss<dim - 1>             quadrature(1);
      std::vector<double>         pressure_values(1);
      FEFaceValues<dim>           fe_values(this->mapping,
                                  fe_p,
                                  quadrature,
                                  update_values | update_quadrature_points);
      for (typename DoFHandler<dim>::active_cell_iterator cell =
             dof_handler_p.begin_active();
           cell != dof_handler_p.end();
           ++cell)
        if (cell->is_locally_owned())
          for (unsigned int face = 0; face < cell->n_faces(); ++face)
            if (cell->at_boundary(face))
              {
                typename std::map<types::boundary_id,
                                  std::shared_ptr<Function<dim>>>::iterator it =
                  this->boundary->pressure_fix.find(cell->face(face)->boundary_id());
                if (it != this->boundary->pressure_fix.end())
                  {
                    fe_values.reinit(cell, face);
                    fe_values.get_function_values(solution.block(1), pressure_values);
                    cell->get_dof_indices(dof_indices);
                    distance.first = dof_indices.back();
                    if (it->second.get() != 0)
                      {
                        it->second->set_time(time_stepping.now());
                        distance.second =
                          it->second->value(fe_values.quadrature_point(0)) -
                          pressure_values[0];
                      }
                    else
                      distance.second = -pressure_values[0];

                    goto end_loop;
                  }
              }
    end_loop:

      const long int min_index =
        -Utilities::MPI::max(-distance.first, get_communicator(triangulation));
      AssertThrow(min_index != std::numeric_limits<long int>::max(),
                  ExcMessage("Could not find a boundary point for fixing the pressure"));
      const double local_value =
        (distance.first == min_index ? distance.second :
                                       -std::numeric_limits<double>::max());
      const double shift =
        Utilities::MPI::max(local_value, get_communicator(triangulation));
      navier_stokes_matrix.apply_pressure_shift(shift, solution.block(1));
      hanging_node_constraints_p.distribute(solution.block(1));
      solution.block(1).update_ghost_values();
    }

  // For projection, we have Dirichlet conditions set on the 'pressure'
  // boundaries, so need to adjust the pressure value here...
  if (!this->boundary->open_conditions_p.empty() &&
      this->parameters.linearization == FlowParameters::projection)
    {
      std::map<types::boundary_id, const Function<dim> *> dirichlet_p;

      // write the prescribed functions as Dirichlet values into the pressure
      for (typename std::map<types::boundary_id, std::shared_ptr<Function<dim>>>::iterator
             it = this->boundary->open_conditions_p.begin();
           it != this->boundary->open_conditions_p.end();
           ++it)
        {
          it->second->set_time(this->time_stepping.now());
          dirichlet_p[it->first] = it->second.get();
        }

      std::map<types::global_dof_index, double> boundary_values;
      VectorTools::interpolate_boundary_values(this->mapping,
                                               dof_handler_p,
                                               dirichlet_p,
                                               boundary_values);

      for (typename std::map<types::global_dof_index, double>::const_iterator it =
             boundary_values.begin();
           it != boundary_values.end();
           ++it)
        if (solution.block(1).in_local_range(it->first))
          solution.block(1)(it->first) = it->second;
      hanging_node_constraints_p.distribute(solution.block(1));
    }

  solver_timers[1].second += nl_timer.wall_time();
  solver_timers[1].first++;

  if ((parameters.output_verbosity > 1 && time_stepping.step_no() % 10 == 1) ||
      parameters.output_verbosity == 3)
    {
      std::ios_base::fmtflags flags = std::cout.flags();
      pcout
        << "-- Statistics --                    min      avg      max avg/call  p_min  p_max"
        << std::endl;
      Utilities::System::MemoryStats stats;
      Utilities::System::get_memory_stats(stats);
      Utilities::MPI::MinMaxAvg memory =
        Utilities::MPI::min_max_avg(stats.VmRSS / 1024, get_communicator(triangulation));
      pcout << "-- Statistics -- memory [MB] : " << std::fixed << std::setprecision(0)
            << std::right << std::setw(8) << memory.min << " " << std::setprecision(0)
            << std::right << std::setw(8) << memory.avg << " " << std::setprecision(0)
            << std::right << std::setw(8) << memory.max << "           " << std::setw(6)
            << std::left << memory.min_index << " " << std::setw(6) << std::left
            << memory.max_index << std::endl;

      std::cout.unsetf(std::ios_base::floatfield);

      memory = Utilities::MPI::min_max_avg(solver_timers[1].second,
                                           get_communicator(triangulation));
      pcout << "-- Statistics -- nln solver  : " << std::setprecision(3) << std::right
            << std::setw(8) << memory.min << " " << std::setprecision(3) << std::right
            << std::setw(8) << memory.avg << " " << std::setprecision(3) << std::right
            << std::setw(8) << memory.max << " " << std::setprecision(3) << std::right
            << std::setw(8) << memory.avg / solver_timers[1].first << "  " << std::setw(6)
            << std::left << memory.min_index << " " << std::setw(6) << std::left
            << memory.max_index << std::endl;
      solver_timers[1] = std::pair<unsigned int, double>();

      memory = Utilities::MPI::min_max_avg(solver_timers[0].second,
                                           get_communicator(triangulation));
      pcout << "-- Statistics --  lin solver : " << std::setprecision(3) << std::right
            << std::setw(8) << memory.min << " " << std::setprecision(3) << std::right
            << std::setw(8) << memory.avg << " " << std::setprecision(3) << std::right
            << std::setw(8) << memory.max << " " << std::setprecision(3) << std::right
            << std::setw(8) << memory.avg / solver_timers[0].first << "  " << std::setw(6)
            << std::left << memory.min_index << " " << std::setw(6) << std::left
            << memory.max_index << std::endl;
      solver_timers[0] = std::pair<unsigned int, double>();

      std::pair<Utilities::MPI::MinMaxAvg, unsigned int> matvec_time =
        navier_stokes_matrix.get_matvec_statistics();
      pcout << "-- Statistics --   mat-vec   : " << std::setprecision(3) << std::right
            << std::setw(8) << matvec_time.first.min << " " << std::setprecision(3)
            << std::right << std::setw(8) << matvec_time.first.avg << " "
            << std::setprecision(3) << std::right << std::setw(8) << matvec_time.first.max
            << " " << std::setprecision(3) << std::right << std::setw(8)
            << matvec_time.first.avg / matvec_time.second << "  " << std::setw(6)
            << std::left << matvec_time.first.min_index << " " << std::setw(6)
            << std::left << matvec_time.first.max_index << std::endl;

      std::string names[5] = {
        "velocity", "div matrix", "pres mass", "pres Poiss", "full prec"};
      std::pair<Utilities::MPI::MinMaxAvg[5], unsigned int> prec_time =
        preconditioner.get_timer_statistics();
      for (unsigned int i = 0; i < 5; ++i)
        {
          const unsigned int ind = (4 + i) % 5;
          pcout << "-- Statistics --   " << std::setw(10) << names[ind] << ": "
                << std::setprecision(3) << std::right << std::setw(8)
                << prec_time.first[ind].min << " " << std::setprecision(3) << std::right
                << std::setw(8) << prec_time.first[ind].avg << " " << std::setprecision(3)
                << std::right << std::setw(8) << prec_time.first[ind].max << " "
                << std::setprecision(3) << std::right << std::setw(8)
                << prec_time.first[ind].avg / prec_time.second << "  " << std::setw(6)
                << std::left << prec_time.first[ind].min_index << " " << std::setw(6)
                << std::left << prec_time.first[ind].max_index << std::endl;
        }
      pcout << std::endl;
      std::cout.flags(flags);
    }

  return step;
}


template <int dim>
void
NavierStokes<dim>::compute_initial_stokes_field()
{
  apply_boundary_conditions();
  if (solution.block(0).l2_norm() > 0)
    {
      // set problem type to Stokes and use AMG preconditioner in any case
      const FlowParameters::PhysicalType         physical_type = parameters.physical_type;
      const FlowParameters::PreconditionVelocity precondition_type =
        parameters.precondition_velocity;
      parameters.precondition_velocity = FlowParameters::u_amg_linear;
      update_preconditioner            = true;
      parameters.physical_type         = FlowParameters::stokes;
      const double density             = parameters.density;
      parameters.density               = 0;
      if (navier_stokes_matrix.use_variable_coefficients() == true)
        {
          const VectorizedArray<double> viscosity =
            make_vectorized_array(parameters.viscosity);
          for (unsigned int cell = 0; cell < matrix_free->n_cell_batches(); ++cell)
            {
              VectorizedArray<double> *visc =
                navier_stokes_matrix.begin_viscosities(cell);
              VectorizedArray<double> *dens = navier_stokes_matrix.begin_densities(cell);
              for (unsigned int q = 0; q < matrix_free->get_n_q_points(quad_index_u); ++q)
                {
                  visc[q] = viscosity;
                  dens[q] = VectorizedArray<double>();
                }
            }
        }

      if (parameters.output_verbosity > 0)
        pcout << "  Compute initial velocity field with Stokes" << std::endl;
      const double res = compute_initial_residual(false);
      solve_nonlinear_system(res);


      // reset parameters
      parameters.physical_type         = physical_type;
      parameters.density               = density;
      update_preconditioner            = true;
      parameters.precondition_velocity = precondition_type;
    }
}



template <int dim>
void
NavierStokes<dim>::apply_boundary_conditions()
{
  const double time = time_stepping.now();
  {
    std::map<types::boundary_id, const Function<dim> *> dirichlet_u;

    // evaluate Dirichlet boundaries and no-slip boundaries
    // modify the time in the function
    for (typename std::map<types::boundary_id, std::shared_ptr<Function<dim>>>::iterator
           it = this->boundary->dirichlet_conditions_u.begin();
         it != this->boundary->dirichlet_conditions_u.end();
         ++it)
      {
        it->second->set_time(time);
        dirichlet_u[it->first] = it->second.get();
      }

    Functions::ZeroFunction<dim> zero_func(dim);
    for (typename std::set<types::boundary_id>::const_iterator it =
           this->boundary->no_slip.begin();
         it != this->boundary->no_slip.end();
         ++it)
      dirichlet_u[*it] = &zero_func;

    std::map<types::global_dof_index, double> boundary_values;
    VectorTools::interpolate_boundary_values(this->mapping,
                                             dof_handler_u,
                                             dirichlet_u,
                                             boundary_values);

    for (typename std::map<types::global_dof_index, double>::const_iterator it =
           boundary_values.begin();
         it != boundary_values.end();
         ++it)
      if (solution.block(0).in_local_range(it->first))
        solution.block(0)(it->first) = it->second;
    hanging_node_constraints_u.distribute(solution.block(0));
  }

  solution.update_ghost_values();
  solution_old.update_ghost_values();
  solution_old_old.update_ghost_values();

  const_rhs = 0;
  if (!this->boundary->open_conditions_p.empty())
    {
      for (typename std::map<types::boundary_id, std::shared_ptr<Function<dim>>>::iterator
             it = this->boundary->open_conditions_p.begin();
           it != this->boundary->open_conditions_p.end();
           ++it)
        it->second->set_time(time);

      Quadrature<dim - 1> face_quadrature;

      if (parameters.use_simplex_mesh)
        face_quadrature = QGaussSimplex<dim - 1>(fe_u.degree + 1);
      else
        face_quadrature = QGauss<dim - 1>(fe_u.degree + 1);

      FEFaceValues<dim> fe_values(this->mapping,
                                  fe_u,
                                  face_quadrature,
                                  update_values | update_JxW_values |
                                    update_quadrature_points | update_normal_vectors);

      std::vector<double>                        boundary_values(face_quadrature.size());
      LinearAlgebra::distributed::Vector<double> cell_rhs(fe_u.dofs_per_cell);
      std::vector<types::global_dof_index>       dof_indices(fe_u.dofs_per_cell);
      const FEValuesExtractors::Vector           velocities(0);

      for (typename DoFHandler<dim>::active_cell_iterator cell =
             dof_handler_u.begin_active();
           cell != dof_handler_u.end();
           ++cell)
        if (cell->is_locally_owned())
          for (unsigned int face = 0; face < cell->n_faces(); ++face)
            if (cell->at_boundary(face) && (this->boundary->open_conditions_p.find(
                                              cell->face(face)->boundary_id()) !=
                                            this->boundary->open_conditions_p.end()))
              {
                fe_values.reinit(cell, face);
                this->boundary->open_conditions_p.find(cell->face(face)->boundary_id())
                  ->second->value_list(fe_values.get_quadrature_points(),
                                       boundary_values);

                for (unsigned int i = 0; i < fe_u.dofs_per_cell; ++i)
                  {
                    double value = 0;
                    for (unsigned int q = 0; q < face_quadrature.size(); ++q)
                      value += ((fe_values[velocities].value(i, q) *
                                 fe_values.normal_vector(q)) *
                                boundary_values[q] * fe_values.JxW(q));
                    cell_rhs(i) = value;
                  }

                cell->get_dof_indices(dof_indices);
                constraints_u.distribute_local_to_global(cell_rhs,
                                                         dof_indices,
                                                         const_rhs);
              }
    }
  const_rhs.compress(VectorOperation::add);
}



template <int dim>
void
NavierStokes<dim>::refine_grid_pressure_based(const unsigned int max_grid_level,
                                              const double       refine_fraction_of_cells,
                                              const double coarsen_fraction_of_cells)
{
  // The Kelly estimator needs all degrees of freedom on neighboring cells
  // because it computes the jumps of gradients over faces. The solution
  // vector does not have all these degrees of freedom imported locally, so
  // need to get a vector with extended ghosting.
  LinearAlgebra::distributed::Vector<double> pressure_extended(
    dof_handler_p.locally_owned_dofs(),
    constraints_p.get_local_lines(),
    get_communicator(triangulation));
  pressure_extended = solution.block(1);
  pressure_extended.update_ghost_values();
  Vector<float> estimated_error_per_cell(triangulation.n_active_cells());
  KellyErrorEstimator<dim>::estimate(
    dof_handler_p,
    QGauss<dim - 1>(parameters.velocity_degree + 2),
    std::map<types::boundary_id, const Function<dim> *>(),
    pressure_extended,
    estimated_error_per_cell,
    ComponentMask(),
    0,
    -1,
    this_mpi_process);

  auto tria = dynamic_cast<parallel::distributed::Triangulation<dim> *>(&triangulation);

  Assert(tria, ExcNotImplemented());

  parallel::distributed::GridRefinement::refine_and_coarsen_fixed_number(
    *tria, estimated_error_per_cell, refine_fraction_of_cells, coarsen_fraction_of_cells);

  if (triangulation.n_levels() > max_grid_level)
    for (typename Triangulation<dim>::active_cell_iterator cell =
           triangulation.begin_active(max_grid_level);
         cell != triangulation.end();
         ++cell)
      if (cell->is_locally_owned())
        cell->clear_refine_flag();

  prepare_coarsening_and_refinement();
  triangulation.execute_coarsening_and_refinement();
  distribute_dofs();
}



template <int dim>
void
NavierStokes<dim>::prepare_coarsening_and_refinement()
{
  sol_trans_u =
    std::make_shared<parallel::distributed::
                       SolutionTransfer<dim, LinearAlgebra::distributed::Vector<double>>>(
      dof_handler_u);
  sol_trans_p =
    std::make_shared<parallel::distributed::
                       SolutionTransfer<dim, LinearAlgebra::distributed::Vector<double>>>(
      dof_handler_p);

  hanging_node_constraints_u.distribute(solution.block(0));
  hanging_node_constraints_u.distribute(solution_old.block(0));
  hanging_node_constraints_p.distribute(solution.block(1));
  hanging_node_constraints_p.distribute(solution_old.block(1));
  solution.update_ghost_values();
  solution_old.update_ghost_values();

  triangulation.prepare_coarsening_and_refinement();

  std::vector<const LinearAlgebra::distributed::Vector<double> *> old_grid_solutions(2);
  old_grid_solutions[0] = &solution.block(0);
  old_grid_solutions[1] = &solution_old.block(0);
  sol_trans_u->prepare_for_coarsening_and_refinement(old_grid_solutions);

  old_grid_solutions[0] = &solution.block(1);
  old_grid_solutions[1] = &solution_old.block(1);
  // Need this vector for projection algorithm, so transfer it here.
  if (parameters.linearization == FlowParameters::projection)
    {
      hanging_node_constraints_p.distribute(solution_old_old.block(1));
      solution_old_old.block(1).update_ghost_values();
      old_grid_solutions.push_back(&solution_old_old.block(1));
    }
  sol_trans_p->prepare_for_coarsening_and_refinement(old_grid_solutions);
}



template <int dim>
void
NavierStokes<dim>::interpolate_pressure_field(
  const Function<dim> &                       pressure_function,
  LinearAlgebra::distributed::Vector<double> &pressure_vector) const
{
  VectorTools::interpolate(dof_handler_p, pressure_function, pressure_vector);
  if (parameters.augmented_taylor_hood)
    {
      // set DG0 components to zero
      std::vector<std::vector<bool>> constant_modes;
      DoFTools::extract_constant_modes(dof_handler_p,
                                       std::vector<bool>(1, true),
                                       constant_modes);
      AssertDimension(constant_modes.size(), 2);
      for (unsigned int i = 0; i < pressure_vector.local_size(); ++i)
        if (constant_modes[1][i])
          pressure_vector.local_element(i) = 0.;
    }
}


template <int dim>
void
NavierStokes<dim>::output_solution(const std::string  output_name,
                                   const unsigned int n_subdivisions) const
{
  DataOut<dim> data_out;

  std::vector<DataComponentInterpretation::DataComponentInterpretation>
    vector_component_interpretation(
      dim, DataComponentInterpretation::component_is_part_of_vector);

  solution.update_ghost_values();

  data_out.add_data_vector(get_dof_handler_u(),
                           solution.block(0),
                           std::vector<std::string>(dim, "velocity"),
                           vector_component_interpretation);
  data_out.add_data_vector(get_dof_handler_p(), solution.block(1), "pressure");
  const unsigned int n_patches = n_subdivisions == 0 ? fe_u.degree : n_subdivisions;
  data_out.build_patches(this->mapping, n_patches);

  this->write_data_output(output_name,
                          time_stepping,
                          parameters.output_frequency,
                          this->triangulation,
                          data_out);
}



template <int dim>
std::size_t
NavierStokes<dim>::memory_consumption() const
{
  std::size_t memory = sizeof(this);
  memory += dof_handler_u.memory_consumption() + dof_handler_p.memory_consumption();
  memory += fe_u.memory_consumption() + fe_p.memory_consumption();
  memory += constraints_u.memory_consumption() + constraints_p.memory_consumption() +
            hanging_node_constraints_u.memory_consumption() +
            hanging_node_constraints_p.memory_consumption();
  memory += navier_stokes_matrix.memory_consumption();
  memory += preconditioner.memory_consumption();
  memory += matrix_free->memory_consumption();
  memory += 6 * system_rhs.memory_consumption();
  return memory;
}



template <int dim>
void
NavierStokes<dim>::print_memory_consumption(std::ostream &stream) const
{
  if (this_mpi_process == 0)
    {
      stream << "\n+--- Memory consumption Navier-Stokes objects ---\n"
             << "| DoFHandler: "
             << 1e-6 * double(dof_handler_u.memory_consumption() +
                              dof_handler_p.memory_consumption())
             << " MB\n"
             //<< "FE: "
             //   << 1e-6*double(fe_u.memory_consumption()+
             //      fe_p.memory_consumption())
             // << " MB\n"
             << "| Matrix-free objects: "
             << 1e-6 * double(matrix_free->memory_consumption()) << " MB\n"
             << "| Constraints: "
             << 1e-6 * double(constraints_u.memory_consumption() +
                              constraints_p.memory_consumption() +
                              hanging_node_constraints_u.memory_consumption() +
                              hanging_node_constraints_p.memory_consumption())
             << " MB\n";
      navier_stokes_matrix.print_memory_consumption(stream);
      preconditioner.print_memory_consumption(stream);
      stream << "| Vectors: " << 6e-6 * double(system_rhs.memory_consumption())
             << " MB\n";
      stream << "| Memory consumption used objects:\n"
             << "| Triangulation: " << 1e-6 * double(triangulation.memory_consumption())
             << " MB\n"
             << "+------------------------------------------------\n";
    }
}


// explicit instantiations
template class NavierStokes<2>;
template class NavierStokes<3>;
