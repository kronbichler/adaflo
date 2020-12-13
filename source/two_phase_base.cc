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

#include <deal.II/base/utilities.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_fe.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/lac/lapack_full_matrix.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/simplex/fe_lib.h>
#include <deal.II/simplex/quadrature_lib.h>

#include <adaflo/two_phase_base.h>
#include <adaflo/util.h>

#include <fstream>
#include <iostream>


using namespace dealii;


namespace
{
  template <int dim>
  std::vector<Point<dim>>
  get_unit_cell_face_centers()
  {
    std::vector<Point<dim>> face_centers;
    Triangulation<dim>      tria;
    GridGenerator::hyper_cube(tria, 0, 1);
    for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
      face_centers.push_back(tria.begin()->face(f)->center());
    return face_centers;
  }
} // namespace



template <int dim>
TwoPhaseBaseAlgorithm<dim>::TwoPhaseBaseAlgorithm(
  const FlowParameters &                    parameters_in,
  const std::shared_ptr<FiniteElement<dim>> fe_in,
  Triangulation<dim> &                      tria_in,
  TimerOutput *                             timer_in)
  : FlowBaseAlgorithm<dim>(
      parameters_in.use_simplex_mesh ?
        std::shared_ptr<Mapping<dim>>(new MappingFE<dim>(Simplex::FE_P<dim>(1))) :
        std::shared_ptr<Mapping<dim>>(new MappingQ<dim>(3)))
  , solution_update(2)
  , solution(2)
  , solution_old(2)
  , solution_old_old(2)
  , pcout(std::cout, Utilities::MPI::this_mpi_process(get_communicator(tria_in)) == 0)
  , timer((timer_in == 0 ?
             new TimerOutput(pcout,
                             parameters_in.output_wall_times ? TimerOutput::summary :
                                                               TimerOutput::never,
                             TimerOutput::wall_times) :
             timer_in),
          helpers::DummyDeleter<TimerOutput>(timer_in == 0))
  , triangulation(tria_in)
  , navier_stokes(parameters_in, triangulation, timer.get(), this->boundary)
  , fe(fe_in)
  , dof_handler(triangulation)
  , system_rhs(2)
  , time_stepping(navier_stokes.time_stepping)
  , parameters(navier_stokes.get_parameters())
  , epsilon_used(0)
  , minimal_edge_length(0)
  , face_center_quadrature(get_unit_cell_face_centers<dim>())
  , curvature_name("dummy_curvature")
{}



template <int dim>
TwoPhaseBaseAlgorithm<dim>::~TwoPhaseBaseAlgorithm()
{
  matrix_free.clear();
  dof_handler.clear();
  std::cout.unsetf(std::ios_base::floatfield);
}



template <int dim>
void
TwoPhaseBaseAlgorithm<dim>::clear_data()
{}



template <int dim>
void
TwoPhaseBaseAlgorithm<dim>::setup_problem(const Function<dim> &initial_velocity_field,
                                          const Function<dim> &initial_distance_function)
{
  timer->enter_subsection("TP setup problem.");

  global_omega_diameter = GridTools::diameter(triangulation);
  // if we should to more than 15 refinements, this can't be right: We would
  // get 1e9 as many elements in 2d and 3e13 in 3d! The user likely used this
  // variables for specifying how often to refine a rectangle...
  if (parameters.global_refinements < 15)
    triangulation.refine_global(parameters.global_refinements);

  navier_stokes.time_stepping.restart();
  distribute_dofs();

  initialize_data_structures();

  // apply initial condition
  if (!time_stepping.at_end())
    {
      VectorTools::interpolate(this->mapping,
                               navier_stokes.get_dof_handler_u(),
                               initial_velocity_field,
                               navier_stokes.solution.block(0));
      navier_stokes.solution.update_ghost_values();
      navier_stokes.solution_old.update_ghost_values();
    }

  VectorTools::interpolate(this->mapping,
                           dof_handler,
                           initial_distance_function,
                           solution.block(0));

  transform_distance_function(solution.block(0));
  solution.update_ghost_values();
  solution_old.update_ghost_values();

  refine_lower_level_limit = triangulation.n_levels() - 1;

  {
    unsigned int mesh_adaptation =
      parameters.adaptive_refinements > 0 ? parameters.adaptive_refinements + 1 : 0;
    while (mesh_adaptation > 0)
      {
        refine_grid();

        navier_stokes.solution.block(0).zero_out_ghosts();
        VectorTools::interpolate(this->mapping,
                                 navier_stokes.get_dof_handler_u(),
                                 initial_velocity_field,
                                 navier_stokes.solution.block(0));
        navier_stokes.solution.update_ghost_values();
        navier_stokes.solution_old.update_ghost_values();

        solution.block(0).zero_out_ghosts();
        VectorTools::interpolate(this->mapping,
                                 dof_handler,
                                 initial_distance_function,
                                 solution.block(0));
        transform_distance_function(solution.block(0));
        hanging_node_constraints.distribute(solution.block(0));
        solution_old.update_ghost_values();
        solution.update_ghost_values();

        mesh_adaptation--;
      }
  }

  // compute divergence-free velocity field in case we start from zero
  // velocity but the boundary condition says something else
  if (this->navier_stokes.solution.block(0).l2_norm() == 0)
    navier_stokes.compute_initial_stokes_field();

  timer->leave_subsection();
}



template <int dim>
void
TwoPhaseBaseAlgorithm<dim>::distribute_dofs()
{
  clear_data();
  constraints.clear();
  constraints_curvature.clear();
  hanging_node_constraints.clear();
  constraints_normals.clear();

  navier_stokes.distribute_dofs();
  dof_handler.distribute_dofs(*fe);

  IndexSet relevant_dofs;
  DoFTools::extract_locally_relevant_dofs(dof_handler, relevant_dofs);
  constraints.reinit(relevant_dofs);
  constraints_curvature.reinit(relevant_dofs);
  hanging_node_constraints.reinit(relevant_dofs);
  constraints_normals.reinit(relevant_dofs);

  DoFTools::make_hanging_node_constraints(dof_handler, hanging_node_constraints);
  constraints.merge(hanging_node_constraints);
  constraints_curvature.merge(hanging_node_constraints);
  constraints_normals.merge(hanging_node_constraints);
}



template <int dim>
void
TwoPhaseBaseAlgorithm<dim>::initialize_data_structures()
{
  hanging_node_constraints.close();
  constraints.close();
  constraints_curvature.close();
  constraints_normals.close();

  navier_stokes.initialize_data_structures();

  typename MatrixFree<dim>::AdditionalData data;
  data.tasks_parallel_scheme = MatrixFree<dim>::AdditionalData::partition_partition;
  data.store_plain_indices   = true;
  std::vector<const DoFHandler<dim> *> dof_handlers;
  dof_handlers.push_back(&navier_stokes.get_dof_handler_u());
  dof_handlers.push_back(&navier_stokes.get_dof_handler_p());
  dof_handlers.push_back(&dof_handler);
  dof_handlers.push_back(&dof_handler);
  dof_handlers.push_back(&dof_handler);

  std::vector<const AffineConstraints<double> *> constraint;
  constraint.push_back(&navier_stokes.get_constraints_u());
  constraint.push_back(&navier_stokes.get_constraints_p());
  constraint.push_back(&constraints);
  constraint.push_back(&constraints_curvature);
  constraint.push_back(&constraints_normals);

  std::vector<Quadrature<dim>> quadratures;
  if (parameters.use_simplex_mesh)
    {
      quadratures.push_back(Simplex::QGauss<dim>(parameters.velocity_degree + 1));
      quadratures.push_back(Simplex::QGauss<dim>(parameters.velocity_degree));
      quadratures.push_back(Simplex::QGauss<dim>(fe->degree + 1));
    }
  else
    {
      quadratures.push_back(QGauss<dim>(parameters.velocity_degree + 1));
      quadratures.push_back(QGauss<dim>(parameters.velocity_degree));
      if (fe->get_name().find("FE_Q_iso_Q1") != std::string::npos)
        quadratures.push_back(QIterated<dim>(QGauss<1>(2), fe->degree));
      else
        quadratures.push_back(QGauss<dim>(fe->degree + 1));
    }

  matrix_free.reinit(this->mapping, dof_handlers, constraint, quadratures, data);

  navier_stokes.initialize_matrix_free(&matrix_free);

  print_n_dofs();


  // find relevant epsilon for smoothing by taking the largest mesh size of
  // cells close to the interface (here: cells on finest level)
  epsilon_used        = 0;
  minimal_edge_length = global_omega_diameter;
  cell_diameters.resize(this->matrix_free.n_cell_batches());

  // to find the cell diameters, we compute the maximum and minimum eigenvalue
  // of the Jacobian transformation from the unit to the real cell. We check
  // all face centers and the center of the cell and take the respective
  // minimum and maximum there to cover most of the cell geometry
  std::vector<Point<dim>> face_centers;
  {
    Point<dim> center;
    for (unsigned int d = 0; d < dim; ++d)
      center[d] = 0.5;
    for (unsigned int d = 0; d < dim; ++d)
      {
        Point<dim> p1 = center;
        p1[d]         = 0;
        face_centers.push_back(p1);
        p1[d] = 1.;
        face_centers.push_back(p1);
      }
    face_centers.push_back(center);
  }
  LAPACKFullMatrix<double> mat(dim, dim);
  FEValues<dim>            fe_values(this->mapping,
                          navier_stokes.get_fe_p(),
                          Quadrature<dim>(face_centers),
                          update_jacobians);
  for (unsigned int cell = 0; cell < this->matrix_free.n_cell_batches(); ++cell)
    {
      VectorizedArray<double> diameter = VectorizedArray<double>();
      for (unsigned int v = 0;
           v < this->matrix_free.n_active_entries_per_cell_batch(cell);
           ++v)
        {
          typename DoFHandler<dim>::active_cell_iterator dcell =
            this->matrix_free.get_cell_iterator(cell, v, 1);
          fe_values.reinit(dcell);
          for (unsigned int q = 0; q < fe_values.n_quadrature_points; ++q)
            {
              mat = 0;
              for (unsigned int d = 0; d < dim; ++d)
                for (unsigned int e = 0; e < dim; ++e)
                  mat(d, e) = fe_values.jacobian(q)[d][e];
              mat.compute_eigenvalues();
              for (unsigned int d = 0; d < dim; ++d)
                {
                  diameter[v] = std::max(diameter[v], std::abs(mat.eigenvalue(d)));
                  minimal_edge_length =
                    std::min(minimal_edge_length, std::abs(mat.eigenvalue(d)));
                }
            }
          if (1U + dcell->level() == this->triangulation.n_global_levels())
            epsilon_used = std::max(diameter[v], epsilon_used);
        }
      cell_diameters[cell] = diameter;
    }
  minimal_edge_length =
    -Utilities::MPI::max(-minimal_edge_length, get_communicator(triangulation));
  epsilon_used = Utilities::MPI::max(epsilon_used, get_communicator(triangulation));

  this->pcout << "Mesh size (largest/smallest element length at finest level): "
              << epsilon_used << " / " << minimal_edge_length << std::endl;
  epsilon_used =
    parameters.epsilon / parameters.concentration_subdivisions * epsilon_used;


  // Create two blocks (for concentration and curvature) in the solution
  // vector and right hand side
  solution_update.reinit(2);
  matrix_free.initialize_dof_vector(solution_update.block(0), 2);
  matrix_free.initialize_dof_vector(solution_update.block(1), 3);
  solution_update.collect_sizes();
  solution.reinit(solution_update);
  solution_old.reinit(solution_update);
  solution_old_old.reinit(solution_update);
  system_rhs.reinit(solution_update);
}



template <int dim>
void
TwoPhaseBaseAlgorithm<dim>::print_n_dofs() const
{
  std::pair<unsigned int, unsigned int> ns_dofs = navier_stokes.n_dofs();
  pcout << std::endl
        << "Number of active cells: " << triangulation.n_global_active_cells() << "."
        << std::endl
        << "Number of Navier-Stokes degrees of freedom: "
        << ns_dofs.first + ns_dofs.second << " (" << ns_dofs.first << " + "
        << ns_dofs.second << ")." << std::endl
        << "Number of level set degrees of freedom: " << dof_handler.n_dofs() << "."
        << std::endl;
}



template <int dim>
bool
TwoPhaseBaseAlgorithm<dim>::mark_cells_for_refinement()
{
  // in this base algorithm, refine at most every fifth time step
  if (this->parameters.adaptive_refinements == 0 || time_stepping.step_no() % 5 != 0)
    return false;

  timer->enter_subsection("Probe grid refinement.");
  LinearAlgebra::distributed::Vector<double> error_estimate(solution.block(0));
  Vector<float> error_per_cell(triangulation.n_active_cells());

  {
    for (unsigned int i = 0; i < error_estimate.local_size(); i++)
      error_estimate.local_element(i) =
        (1. - error_estimate.local_element(i) * error_estimate.local_element(i));
    error_estimate.update_ghost_values();
  }

  VectorTools::integrate_difference(dof_handler,
                                    error_estimate,
                                    Functions::ZeroFunction<dim>(2),
                                    error_per_cell,
                                    QGauss<dim>(fe->degree + 1),
                                    VectorTools::L2_norm);

  const double h_to_3 = triangulation.last()->diameter() *
                        triangulation.last()->diameter() *
                        ((dim == 3) ? triangulation.last()->diameter() : 1);
  const int upper_level_limit =
    this->parameters.adaptive_refinements + this->refine_lower_level_limit;

  bool                                           must_refine = false;
  typename DoFHandler<dim>::active_cell_iterator cell        = dof_handler.begin_active(),
                                                 endc        = dof_handler.end();
  for (unsigned int cell_no = 0; cell != endc; ++cell, ++cell_no)
    if (cell->is_locally_owned())
      {
        cell->clear_coarsen_flag();
        cell->clear_refine_flag();

        bool refine_cell = ((cell->level() < upper_level_limit) &&
                            (error_per_cell(cell_no) > 0.3 * h_to_3));

        // do not refine cells at the boundary
        for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face)
          if (cell->face(face)->at_boundary())
            refine_cell = false;
        if (refine_cell == true)
          {
            must_refine = true;
            cell->set_refine_flag();
          }
        else if ((cell->level() > refine_lower_level_limit) &&
                 (error_per_cell(cell_no) < 0.1 * h_to_3))
          {
            must_refine = true;
            cell->set_coarsen_flag();
          }
      }
  const bool global_must_refine =
    Utilities::MPI::max(static_cast<unsigned int>(must_refine),
                        get_communicator(triangulation));
  timer->leave_subsection();
  return global_must_refine;
}



template <int dim>
void
TwoPhaseBaseAlgorithm<dim>::refine_grid()
{
  if (mark_cells_for_refinement() == false)
    return;

  timer->enter_subsection("Refine grid.");

  std::vector<const LinearAlgebra::distributed::Vector<double> *> old_grid_solutions;
  solution_old.update_ghost_values();
  solution.update_ghost_values();
  old_grid_solutions.push_back(&solution_old.block(0));
  old_grid_solutions.push_back(&solution.block(0));
  old_grid_solutions.push_back(&solution_old.block(1));
  old_grid_solutions.push_back(&solution.block(1));

  navier_stokes.prepare_coarsening_and_refinement();

  parallel::distributed::SolutionTransfer<dim, LinearAlgebra::distributed::Vector<double>>
    soltrans(dof_handler);
  soltrans.prepare_for_coarsening_and_refinement(old_grid_solutions);

  triangulation.execute_coarsening_and_refinement();
  dof_handler.clear();

  distribute_dofs();
  initialize_data_structures();

  std::vector<LinearAlgebra::distributed::Vector<double> *> new_grid_solutions;
  new_grid_solutions.push_back(&solution_old.block(0));
  new_grid_solutions.push_back(&solution.block(0));
  new_grid_solutions.push_back(&solution_old.block(1));
  new_grid_solutions.push_back(&solution.block(1));

  soltrans.interpolate(new_grid_solutions);

  hanging_node_constraints.distribute(solution_old);
  hanging_node_constraints.distribute(solution);

  solution_old.update_ghost_values();
  solution.update_ghost_values();

  timer->leave_subsection();
}



template <int dim>
void
TwoPhaseBaseAlgorithm<dim>::init_time_advance()
{
  navier_stokes.init_time_advance(parameters.output_verbosity > 0);

  const double step_size     = time_stepping.step_size();
  const double step_size_old = time_stepping.old_step_size();
  solution_update            = solution;

  if (step_size_old > 0)
    solution_update.sadd((step_size + step_size_old) / step_size_old,
                         -step_size / step_size_old,
                         solution_old);

  solution_old_old = solution_old;
  solution_old     = solution;
  solution         = solution_update;

  solution.update_ghost_values();
  solution_old.update_ghost_values();
  solution_old_old.update_ghost_values();

  // only do sporadic output of time
  if (parameters.output_verbosity == 0)
    {
      std::cout.precision(3);
      const double frequency = parameters.output_frequency;
      const double time      = time_stepping.now();
      const int    position  = int(time * 1.0000000001 / frequency);
      const double slot      = position * frequency;
      if ((time - slot) < time_stepping.step_size() * 0.95)
        pcout << time << " " << std::flush;
    }
}



template <int dim>
double
TwoPhaseBaseAlgorithm<dim>::get_maximal_velocity() const
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



template <int dim>
std::pair<double, double>
TwoPhaseBaseAlgorithm<dim>::get_concentration_range() const
{
  const QIterated<dim> quadrature_formula(QTrapez<1>(), fe->degree + 2);
  FEValues<dim>        fe_values(*fe, quadrature_formula, update_values);
  const unsigned int   n_q_points = quadrature_formula.size();
  std::vector<double>  concentration_values(n_q_points);

  double min_concentration = std::numeric_limits<double>::max(),
         max_concentration = -min_concentration;

  typename DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active(),
                                                 endc = dof_handler.end();
  for (; cell != endc; ++cell)
    if (cell->is_locally_owned())
      {
        fe_values.reinit(cell);
        fe_values.get_function_values(solution.block(0), concentration_values);

        for (unsigned int q = 0; q < n_q_points; ++q)
          {
            const double concentration = concentration_values[q];

            min_concentration = std::min(min_concentration, concentration);
            max_concentration = std::max(max_concentration, concentration);
          }
      }
  last_concentration_range = std::make_pair(
    -Utilities::MPI::max(-min_concentration, get_communicator(triangulation)),
    Utilities::MPI::max(max_concentration, get_communicator(triangulation)));
  return last_concentration_range;
}



// @sect4{TwoPhaseBaseAlgorithm::output_solution}
template <int dim>
void
TwoPhaseBaseAlgorithm<dim>::output_solution(const std::string  output_name,
                                            const unsigned int n_subdivisions) const
{
  if (time_stepping.at_tick(parameters.output_frequency) == false)
    return;

  if (parameters.print_solution_fields == false)
    return;

  AssertThrow(output_name.length() > 0, ExcMessage("No valid filename given"));
  timer->enter_subsection("TP create output.");

  DataOut<dim> data_out;

  data_out.add_data_vector(
    navier_stokes.get_dof_handler_u(),
    navier_stokes.solution.block(0),
    std::vector<std::string>(dim, "velocity"),
    std::vector<DataComponentInterpretation::DataComponentInterpretation>(
      dim, DataComponentInterpretation::component_is_part_of_vector));
  data_out.add_data_vector(navier_stokes.get_dof_handler_p(),
                           navier_stokes.solution.block(1),
                           "pressure");
  data_out.add_data_vector(dof_handler, solution.block(0), "concentration");
  data_out.add_data_vector(dof_handler, solution.block(1), curvature_name);

  const unsigned int n_patches =
    n_subdivisions == 0 ?
      std::min(parameters.velocity_degree, parameters.concentration_subdivisions) :
      n_subdivisions;
  data_out.build_patches(this->mapping, n_patches);

  this->write_data_output(output_name,
                          this->time_stepping,
                          this->parameters.output_frequency,
                          this->triangulation,
                          data_out);

  timer->leave_subsection();
}



template <int dim>
void
TwoPhaseBaseAlgorithm<dim>::set_adaptive_time_step(const double norm_velocity) const
{
  // Evaluate the time step according to the stability condition.

  const double cfl       = parameters.time_stepping_cfl;
  const double rho_2     = parameters.viscosity_diff + parameters.viscosity;
  const double coef_2    = parameters.time_stepping_coef2;
  const double sigma_val = parameters.surface_tension;

  double new_time_step =
    1 /
    (1 / (cfl * minimal_edge_length / norm_velocity) +
     1 / (coef_2 * std::sqrt(rho_2 / sigma_val) * std::pow(minimal_edge_length, 1.5)));

  // hand this step to the timer stepper. The time stepper will make sure that
  // the time step does not change too rapidly from one iteration to the next
  // and also be within the bounds set in the parameter file.
  time_stepping.set_time_step(new_time_step);
}



template <>
std::vector<double> TwoPhaseBaseAlgorithm<2>::compute_bubble_statistics(
  std::vector<Tensor<2, 2>> *interface_points,
  const unsigned int         sub_refinements) const
{
  const unsigned int dim = 2;

  const int sub_per_d = sub_refinements == numbers::invalid_unsigned_int ?
                          parameters.velocity_degree + 3 :
                          sub_refinements;
  const QIterated<dim> quadrature_formula(QTrapez<1>(), sub_per_d);
  const QGauss<dim>    interior_quadrature(parameters.velocity_degree);
  const unsigned int   n_q_points = quadrature_formula.size();
  FEValues<dim>        fe_values(this->mapping,
                          *fe,
                          quadrature_formula,
                          update_values | update_JxW_values | update_quadrature_points);
  FEValues<dim>        ns_values(this->mapping,
                          navier_stokes.get_fe_u(),
                          quadrature_formula,
                          update_values);
  FEValues<dim>        interior_ns_values(this->mapping,
                                   navier_stokes.get_fe_u(),
                                   interior_quadrature,
                                   update_values | update_JxW_values |
                                     update_quadrature_points);

  const FEValuesExtractors::Vector vel(0);

  const unsigned int n_points       = 2 * (dim > 1 ? 2 : 1) * (dim > 2 ? 2 : 1),
                     n_subdivisions = (sub_per_d) * (dim > 1 ? (sub_per_d) : 1) *
                                      (dim > 2 ? (sub_per_d) : 1);
  std::vector<double> full_c_values(n_q_points), c_values(n_points),
    quad_weights(n_points), weight_correction(n_q_points);
  std::vector<Tensor<1, dim>> velocity_values(n_q_points), velocities(n_points),
    int_velocity_values(interior_quadrature.size());
  std::vector<Point<dim>> quad(n_points);
  Vector<double>          sol_values(fe->dofs_per_cell);
  for (unsigned int i = 0; i < n_q_points; i++)
    {
      weight_correction[i] = 1;
      unsigned int fact    = sub_per_d + 1;
      if (i % fact > 0 && i % fact < fact - 1)
        weight_correction[i] *= 0.5;
      if (i >= fact && i < n_q_points - fact)
        weight_correction[i] *= 0.5;
    }

  if (interface_points != 0)
    interface_points->clear();
  double                                area = 0, perimeter = 0;
  Tensor<1, dim>                        center_of_mass, velocity;
  DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active(),
                                        endc = dof_handler.end();
  DoFHandler<dim>::active_cell_iterator ns_cell =
    navier_stokes.get_dof_handler_u().begin_active();
  for (; cell != endc; ++cell, ++ns_cell)
    if (cell->is_locally_owned())
      {
        // cheap test: find out whether the interface crosses this cell,
        // i.e. two solution values have a different sign. if not, can compute
        // with a low order Gauss quadrature without caring about the interface
        cell->get_interpolated_dof_values(solution.block(0), sol_values);
        bool interface_crosses_cell = false;
        for (unsigned int i = 1; i < fe->dofs_per_cell; ++i)
          if (sol_values(i) * sol_values(0) <= 0)
            interface_crosses_cell = true;

        if (interface_crosses_cell == false)
          {
            bool has_area = sol_values(0) > 0;
            interior_ns_values.reinit(ns_cell);
            interior_ns_values[vel].get_function_values(navier_stokes.solution.block(0),
                                                        int_velocity_values);
            for (unsigned int q = 0; q < interior_quadrature.size(); q++)
              {
                if (has_area)
                  {
                    area += interior_ns_values.JxW(q);
                    for (unsigned int d = 0; d < dim; ++d)
                      {
                        center_of_mass[d] += (interior_ns_values.quadrature_point(q)[d] *
                                              interior_ns_values.JxW(q));
                        velocity[d] +=
                          (int_velocity_values[q][d] * interior_ns_values.JxW(q));
                      }
                  }
              }
            continue;
          }

        // when the interface crosses this cell, have to find the crossing
        // points (linear interpolation) and compute the area fraction
        fe_values.reinit(cell);
        fe_values.get_function_values(solution.block(0), full_c_values);
        ns_values.reinit(ns_cell);
        ns_values[vel].get_function_values(navier_stokes.solution.block(0),
                                           velocity_values);

        for (unsigned int d = 0; d < n_subdivisions; d++)
          {
            // compute a patch of four points
            {
              const int initial_shift = d % sub_per_d + (d / sub_per_d) * (sub_per_d + 1);
              for (unsigned int i = 0; i < n_points; i++)
                {
                  const unsigned int index =
                    initial_shift + (i / 2) * (sub_per_d - 1) + i;
                  Assert(index < n_q_points, ExcInternalError());
                  c_values[i]     = full_c_values[index];
                  velocities[i]   = velocity_values[index];
                  quad[i]         = fe_values.quadrature_point(index);
                  quad_weights[i] = fe_values.JxW(index) * weight_correction[index];
                }
            }
            double         local_area = 1;
            double         int_rx0 = -1, int_rx1 = -1, int_ry0 = -1, int_ry1 = -1;
            Tensor<1, dim> pos_x0, pos_x1, pos_y0, pos_y1;

            // add a small perturbation to avoid having exact zero values
            for (unsigned int i = 0; i < n_points; ++i)
              c_values[i] += 1e-22;

            // locate interface
            if (c_values[0] * c_values[1] <= 0)
              {
                int_rx0 = c_values[0] / (c_values[0] - c_values[1]);
                pos_x0  = quad[0] + (quad[1] - quad[0]) * int_rx0;
              }
            if (c_values[2] * c_values[3] <= 0)
              {
                int_rx1 = c_values[2] / (c_values[2] - c_values[3]);
                pos_x1  = quad[2] + (quad[3] - quad[2]) * int_rx1;
              }
            if (c_values[0] * c_values[2] <= 0)
              {
                int_ry0 = c_values[0] / (c_values[0] - c_values[2]);
                pos_y0  = quad[0] + (quad[2] - quad[0]) * int_ry0;
              }
            if (c_values[1] * c_values[3] <= 0)
              {
                int_ry1 = c_values[1] / (c_values[1] - c_values[3]);
                pos_y1  = quad[1] + (quad[3] - quad[1]) * int_ry1;
              }
            Tensor<1, dim> difference;
            Tensor<2, dim> interface_p;
            if (int_rx0 > 0)
              {
                if (int_ry0 > 0)
                  {
                    const double my_area = 0.5 * int_rx0 * int_ry0;
                    local_area -= (c_values[0] < 0) ? my_area : 1 - my_area;
                    difference = pos_x0 - pos_y0;
                    perimeter += difference.norm();
                    interface_p[0] = pos_x0;
                    interface_p[1] = pos_y0;
                  }
                if (int_ry1 > 0)
                  {
                    const double my_area = 0.5 * (1 - int_rx0) * int_ry1;
                    local_area -= (c_values[1] < 0) ? my_area : 1 - my_area;
                    difference = pos_x0 - pos_y1;
                    perimeter += difference.norm();
                    interface_p[0] = pos_x0;
                    interface_p[1] = pos_y1;
                  }
                if (int_rx1 > 0 && int_ry0 < 0 && int_ry1 < 0)
                  {
                    const double my_area = 0.5 * (int_rx0 + int_rx1);
                    local_area -= (c_values[0] < 0) ? my_area : 1 - my_area;
                    difference = pos_x0 - pos_x1;
                    perimeter += difference.norm();
                    interface_p[0] = pos_x0;
                    interface_p[1] = pos_x1;
                  }
              }
            if (int_rx1 > 0)
              {
                if (int_ry0 > 0)
                  {
                    const double my_area = 0.5 * int_rx1 * (1 - int_ry0);
                    local_area -= (c_values[2] < 0) ? my_area : 1 - my_area;
                    difference = pos_x1 - pos_y0;
                    perimeter += difference.norm();
                    interface_p[0] = pos_x1;
                    interface_p[1] = pos_y0;
                  }
                if (int_ry1 > 0)
                  {
                    const double my_area = 0.5 * (1 - int_rx1) * (1 - int_ry1);
                    local_area -= (c_values[3] < 0) ? my_area : 1 - my_area;
                    difference = pos_x1 - pos_y1;
                    perimeter += difference.norm();
                    interface_p[0] = pos_x1;
                    interface_p[1] = pos_y1;
                  }
              }
            if (int_ry0 > 0 && int_ry1 > 0 && int_rx0 < 0 && int_rx1 < 0)
              {
                const double my_area = 0.5 * (int_ry0 + int_ry1);
                local_area -= (c_values[0] < 0) ? my_area : 1 - my_area;
                difference = pos_y0 - pos_y1;
                perimeter += difference.norm();
                interface_p[0] = pos_y0;
                interface_p[1] = pos_y1;
              }
            if (int_rx0 <= 0 && int_rx1 <= 0 && int_ry0 <= 0 && int_ry1 <= 0 &&
                c_values[0] <= 0)
              local_area = 0;

            if (interface_p != Tensor<2, dim>() && interface_points != 0)
              interface_points->push_back(interface_p);

            Assert(local_area >= 0, ExcMessage("Subtracted too much"));
            for (unsigned int i = 0; i < n_points; ++i)
              {
                double my_area = local_area * quad_weights[i];
                area += my_area;
                for (unsigned int d = 0; d < dim; ++d)
                  {
                    center_of_mass[d] += quad[i][d] * my_area;
                    velocity[d] += velocities[i][d] * my_area;
                  }
              }
          }
      }

  const MPI_Comm &mpi_communicator = get_communicator(triangulation);

  const double global_area      = Utilities::MPI::sum(area, mpi_communicator);
  const double global_perimeter = Utilities::MPI::sum(perimeter, mpi_communicator);

  Tensor<1, dim> global_mass_center;
  Tensor<1, dim> global_velocity;

  for (unsigned int d = 0; d < dim; ++d)
    {
      global_velocity[d]    = Utilities::MPI::sum(velocity[d], mpi_communicator);
      global_mass_center[d] = Utilities::MPI::sum(center_of_mass[d], mpi_communicator);
    }

  set_adaptive_time_step(global_velocity.norm() / global_area);

  const double circularity = 2. * std::sqrt(global_area * numbers::PI) / global_perimeter;
  if (parameters.output_verbosity > 0)
    {
      const std::size_t old_precision = std::cout.precision();
      std::cout.precision(8);
      pcout << "  Degree of circularity: " << circularity << std::endl;
      pcout << "  Mean bubble velocity: ";
      for (unsigned int d = 0; d < dim; ++d)
        pcout << ((std::abs(global_velocity[d]) < 1e-7 * global_velocity.norm()) ?
                    0. :
                    (global_velocity[d] / global_area))
              << "  ";
      pcout << std::endl;
      pcout << "  Position of the center of mass:  ";
      for (unsigned int d = 0; d < dim; ++d)
        pcout << ((std::abs(global_mass_center[d]) < 1e-7 * this->global_omega_diameter) ?
                    0. :
                    (global_mass_center[d] / global_area))
              << "  ";
      pcout << std::endl;

      std::pair<double, double> concentration = get_concentration_range();
      pcout << "  Range of level set values: " << concentration.first << " / "
            << concentration.second << std::endl;

      std::cout.precision(old_precision);
    }

  std::vector<double> data(4 + 2 * dim);
  data[0] = time_stepping.now();
  data[1] = global_area;
  data[2] = global_perimeter;
  data[3] = circularity;
  for (unsigned int d = 0; d < dim; ++d)
    data[4 + d] = global_velocity[d] / global_area;
  for (unsigned int d = 0; d < dim; ++d)
    data[4 + dim + d] = global_mass_center[d] / global_area;

  // get interface points from other processors
  if (interface_points != 0)
    {
      std::vector<unsigned int> receive_count(
        Utilities::MPI::n_mpi_processes(mpi_communicator));

      unsigned int n_send_elements = interface_points->size();

      MPI_Gather(&n_send_elements,
                 1,
                 MPI_UNSIGNED,
                 &receive_count[0],
                 1,
                 MPI_UNSIGNED,
                 0,
                 mpi_communicator);
      for (unsigned int i = 1; i < Utilities::MPI::n_mpi_processes(mpi_communicator); ++i)
        {
          // Each processor sends the interface_points he deals with to
          // processor
          // 0
          if (Utilities::MPI::this_mpi_process(mpi_communicator) == i)
            {
              // put data into a std::vector<double> to create a data type that
              // MPI understands
              std::vector<double> send_data(2 * dim * interface_points->size());
              for (unsigned int j = 0; j < interface_points->size(); ++j)
                for (unsigned int d = 0; d < 2; ++d)
                  for (unsigned int e = 0; e < dim; ++e)
                    send_data[j * 2 * dim + d * dim + e] = (*interface_points)[j][d][e];
              MPI_Send(
                &send_data[0], send_data.size(), MPI_DOUBLE, 0, i, mpi_communicator);

              // when we are done with sending, destroy the data on all
              // processors except processor 0
              std::vector<Tensor<2, dim>> empty;
              interface_points->swap(empty);
            }

          // Processor 0 receives data from the other processors
          if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
            {
              std::vector<double> receive_data(2 * dim * receive_count[i]);
              int                 ierr = MPI_Recv(&receive_data[0],
                                  receive_data.size(),
                                  MPI_DOUBLE,
                                  i,
                                  i,
                                  mpi_communicator,
                                  MPI_STATUSES_IGNORE);
              (void)ierr;
              Assert(ierr == MPI_SUCCESS, ExcInternalError());
              for (unsigned int j = 0; j < receive_count[i]; ++j)
                {
                  Tensor<2, dim> point;
                  for (unsigned int d = 0; d < 2; ++d)
                    for (unsigned int e = 0; e < dim; ++e)
                      point[d][e] = receive_data[j * 2 * dim + d * dim + e];
                  interface_points->push_back(point);
                }
            }
        }
    }

  return data;
}



template <>
std::vector<double>
  TwoPhaseBaseAlgorithm<3>::compute_bubble_statistics(std::vector<Tensor<2, 3>> *,
                                                      const unsigned int) const
{
  const unsigned int dim = 3;

  const QIterated<dim> quadrature_formula(QGauss<1>(2),
                                          parameters.concentration_subdivisions);

  FEValues<dim> fe_values(this->mapping,
                          *fe,
                          quadrature_formula,
                          update_values | update_JxW_values | update_quadrature_points);

  FEValues<dim> ns_values(this->mapping,
                          navier_stokes.get_fe_u(),
                          quadrature_formula,
                          update_values);

  const unsigned int n_q_points = quadrature_formula.size();

  const FEValuesExtractors::Vector vel(0);

  std::vector<double>         heaviside_values(n_q_points);
  std::vector<Tensor<1, dim>> velocity_values(n_q_points), cell_delta_val(n_q_points);

  double     area = 0, volume = 0;
  Point<dim> center_of_mass, velocity;

  // Here we compute the area of the particle We use by the way the gradient
  // of the level set function. One extract the normal vector

  DoFHandler<dim>::active_cell_iterator ls_cell = dof_handler.begin_active(),
                                        ls_endc = dof_handler.end();
  DoFHandler<dim>::active_cell_iterator ns_cell =
    navier_stokes.get_dof_handler_u().begin_active();


  for (; ls_cell != ls_endc; ++ls_cell, ++ns_cell)
    if (ls_cell->is_locally_owned())
      {
        fe_values.reinit(ls_cell);
        evaluate_heaviside_function(fe_values, heaviside_values, cell_delta_val);
        ns_values.reinit(ns_cell);
        ns_values[vel].get_function_values(navier_stokes.solution.block(0),
                                           velocity_values);

        for (unsigned int q = 0; q < n_q_points; ++q)
          {
            area += 0.5 * cell_delta_val[q].norm() * fe_values.JxW(q);
            volume += heaviside_values[q] * fe_values.JxW(q);

            for (unsigned int d = 0; d < dim; ++d)
              {
                center_of_mass[d] += (fe_values.quadrature_point(q)[d] *
                                      fe_values.JxW(q) * heaviside_values[q]);

                velocity[d] +=
                  (velocity_values[q][d] * fe_values.JxW(q) * heaviside_values[q]);
              }
          }
      }

  const MPI_Comm &mpi_communicator = get_communicator(triangulation);

  Tensor<1, dim> global_mass_center;
  Tensor<1, dim> global_velocity;

  const double global_area   = Utilities::MPI::sum(area, mpi_communicator);
  const double global_volume = Utilities::MPI::sum(volume, mpi_communicator);
  for (unsigned int d = 0; d < dim; ++d)
    {
      global_velocity[d]    = Utilities::MPI::sum(velocity[d], mpi_communicator);
      global_mass_center[d] = Utilities::MPI::sum(center_of_mass[d], mpi_communicator);
    }

  set_adaptive_time_step(global_velocity.norm() / global_volume);

  double pi = numbers::PI;
  if (parameters.output_verbosity > 0)
    {
      const std::size_t old_precision = std::cout.precision();
      std::cout.precision(8);
      pcout << "  Volume of the particle: " << global_volume << std::endl;
      pcout << "  Surface area of the particle: " << global_area << std::endl;
      pcout << "  Mean bubble velocity: ";
      for (unsigned int d = 0; d < dim; ++d)
        pcout << global_velocity[d] / global_volume << "  ";
      pcout << std::endl;
      pcout << "  Position of the center of mass:  ";
      for (unsigned int d = 0; d < dim; ++d)
        pcout << global_mass_center[d] / global_volume << "  ";
      pcout << std::endl;
      pcout << "  Sphericity of the particle: "
            << (std::pow(pi, 1. / 3.) * std::pow(6 * global_volume, 2. / 3.)) /
                 (global_area)
            << std::endl;

      std::cout.precision(10);
      std::pair<double, double> concentration = get_concentration_range();
      pcout << "  Range of level set values: " << concentration.first << " / "
            << concentration.second << std::endl;

      std::cout.precision(old_precision);
    }

  std::vector<double> data(4 + 2 * dim);
  data[0] = time_stepping.now();
  data[1] = global_volume;
  data[2] = global_area;
  for (unsigned int d = 0; d < dim; ++d)
    data[3 + d] = global_velocity[d] / global_volume;
  for (unsigned int d = 0; d < dim; ++d)
    data[3 + dim + d] = global_mass_center[d] / global_volume;
  data[3 + dim + dim] =
    (std::pow(pi, 1. / 3.) * std::pow(6 * global_volume, 2. / 3.)) / (global_area);

  return data;
}



template <int dim>
std::vector<double> TwoPhaseBaseAlgorithm<dim>::compute_bubble_statistics_immersed(
  std::vector<Tensor<2, dim>> * /*interface_points*/) const
{
  // this needs immersed/cut functionality which is not currently available in
  // deal.II

  AssertThrow(false, ExcNotImplemented());
  return std::vector<double>();
}



template class TwoPhaseBaseAlgorithm<2>;
template class TwoPhaseBaseAlgorithm<3>;
