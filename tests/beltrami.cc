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

#include <deal.II/base/function.h>
#include <deal.II/base/function_time.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/utilities.h>

#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <adaflo/navier_stokes.h>
#include <adaflo/time_stepping.h>

#include <fstream>
#include <iomanip>
#include <iostream>

using namespace dealii;



/* ----------------------------------------------------------------------- */

// @sect4{Exact Solution}

// before specifying the initial condition and the boundary conditions, we
// implement the exact solution.  From this, we shall get the necessary
// boundary and initial data afterwards.  The function implemented here is
// the so-called Taylor flow in 2D (see Kim and Moin, JCP 59, pp. 308-323
// (1985)) and the Beltrami flow in 3D (Ethier and Steinman, Int. J. Num.
// Meth. Fluids 19, 369-375, 1994).

template <int dim>
class ExactSolutionU : public Function<dim>
{
public:
  ExactSolutionU(const double viscosity = 1., const double time = 0.)
    : Function<dim>(dim, time)
    , nu(viscosity)
  {}

  virtual void
  vector_value(const Point<dim> &p, Vector<double> &values) const;

private:
  const double nu;
};

template <int dim>
void
ExactSolutionU<dim>::vector_value(const Point<dim> &p, Vector<double> &values) const
{
  AssertDimension(values.size(), dim);

  const double time = this->get_time();

  const double a = 0.25 * numbers::PI;
  const double d = (dim == 3 ? 2. : std::sqrt(2)) * a;

  switch (dim)
    {
      case 3:
        values(0) = -a *
                    (std::exp(a * p[0]) * std::sin(a * p[1] + d * p[2]) +
                     std::exp(a * p[2]) * std::cos(a * p[0] + d * p[1])) *
                    std::exp(-nu * d * d * time);
        values(1) = -a *
                    (std::exp(a * p[1]) * std::sin(a * p[2] + d * p[0]) +
                     std::exp(a * p[0]) * std::cos(a * p[1] + d * p[2])) *
                    std::exp(-nu * d * d * time);
        values(2) = -a *
                    (std::exp(a * p[2]) * std::sin(a * p[0] + d * p[1]) +
                     std::exp(a * p[1]) * std::cos(a * p[2] + d * p[0])) *
                    std::exp(-nu * d * d * time);
        break;
      case 2:
        values(0) = -a * std::cos(a * p[0]) * std::sin(a * p[1]) *
                    std::exp(-2. * nu * a * a * time);
        values(1) =
          a * std::sin(a * p[0]) * std::cos(a * p[1]) * std::exp(-2. * nu * a * a * time);
        break;
      default:
        Assert(false, ExcNotImplemented());
    }
}



template <int dim>
class ExactSolutionP : public Function<dim>
{
public:
  ExactSolutionP(const double viscosity = 1., const double time = 0.)
    : Function<dim>(1, time)
    , nu(viscosity)
  {}

  virtual double
  value(const Point<dim> &p, const unsigned int component) const;

private:
  const double nu;
};



template <int dim>
double
ExactSolutionP<dim>::value(const Point<dim> &p, const unsigned int) const
{
  const double time = this->get_time();

  const double a = 0.25 * numbers::PI;
  const double d = 2. * a;

  double value = 0;
  switch (dim)
    {
      case 3:
        value =
          -a * a * 0.5 *
          (std::exp(2 * a * p[0]) + std::exp(2 * a * p[1]) + std::exp(2 * a * p[2]) +
           2 * std::sin(a * p[0] + d * p[1]) * std::cos(a * p[2] + d * p[0]) *
             std::exp(a * (p[1] + p[2])) +
           2 * std::sin(a * p[1] + d * p[2]) * std::cos(a * p[0] + d * p[1]) *
             std::exp(a * (p[2] + p[0])) +
           2 * std::sin(a * p[2] + d * p[0]) * std::cos(a * p[1] + d * p[2]) *
             std::exp(a * (p[0] + p[1]))) *
          std::exp(-2 * nu * d * d * time);
        break;
      case 2:
        value = -a * a * 0.25 * (std::cos(2 * a * p[0]) + std::cos(2 * a * p[1])) *
                std::exp(-4. * nu * a * a * time);
        break;
      default:
        Assert(false, ExcNotImplemented());
    }
  return value;
}



// @sect3{The <code>Beltramiproblem</code> class template}
template <int dim>
class BeltramiProblem
{
public:
  BeltramiProblem(const FlowParameters &parameters);
  void
  run();

private:
  void
  compute_errors() const;

  void
  output_results() const;

  ConditionalOStream pcout;

  mutable TimerOutput timer;

  parallel::distributed::Triangulation<dim> triangulation;
  NavierStokes<dim>                         navier_stokes;
  const double                              nu;

  mutable LinearAlgebra::distributed::BlockVector<double> exact;
};



template <int dim>
BeltramiProblem<dim>::BeltramiProblem(const FlowParameters &parameters)
  : pcout(std::cout, (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0))
  , timer(pcout, TimerOutput::summary, TimerOutput::cpu_and_wall_times)
  , triangulation(MPI_COMM_WORLD)
  , navier_stokes(parameters, triangulation, &timer)
  , nu(parameters.viscosity)


{}



template <int dim>
void
BeltramiProblem<dim>::compute_errors() const
{
  timer.enter_subsection("Compute errors.");
  navier_stokes.solution.update_ghost_values();

  Vector<float>      cellwise_errors(triangulation.n_active_cells());
  const unsigned int v_degree = navier_stokes.get_parameters().velocity_degree;

  // First compute cell-wise divergence
  QGauss<dim>                quadrature_div(v_degree + 1);
  FEValues<dim>              fe_values(navier_stokes.get_fe_u(),
                          quadrature_div,
                          update_gradients | update_JxW_values);
  std::vector<double>        cell_divergence(quadrature_div.size());
  FEValuesExtractors::Vector velocity(0);
  unsigned int               index = 0;
  for (typename DoFHandler<dim>::active_cell_iterator cell =
         navier_stokes.get_dof_handler_u().begin_active();
       cell != navier_stokes.get_dof_handler_u().end();
       ++cell, ++index)
    if (cell->is_locally_owned())
      {
        fe_values.reinit(cell);
        fe_values[velocity].get_function_divergences(navier_stokes.solution.block(0),
                                                     cell_divergence);
        double div = 0;
        for (unsigned int q = 0; q < quadrature_div.size(); ++q)
          div += cell_divergence[q] * fe_values.JxW(q);
        cellwise_errors(index) = div;
      }
  const double cell_div =
    std::sqrt(Utilities::MPI::sum(cellwise_errors.norm_sqr(), MPI_COMM_WORLD));

  // use high order quadrature to avoid underestimation of errors because of
  // superconvergence effects
  QGauss<dim> quadrature(v_degree + 2);

  // With this, we can then let the library compute the errors and output them
  // to the screen:
  const double time = navier_stokes.time_stepping.now();
  VectorTools::integrate_difference(navier_stokes.get_dof_handler_p(),
                                    navier_stokes.solution.block(1),
                                    ExactSolutionP<dim>(nu, time),
                                    cellwise_errors,
                                    quadrature,
                                    VectorTools::L2_norm);
  const double p_l2_error_loc = cellwise_errors.norm_sqr();
  const double p_l2_error =
    std::sqrt(Utilities::MPI::sum(p_l2_error_loc, MPI_COMM_WORLD));

  VectorTools::integrate_difference(navier_stokes.get_dof_handler_u(),
                                    navier_stokes.solution.block(0),
                                    ExactSolutionU<dim>(nu, time),
                                    cellwise_errors,
                                    quadrature,
                                    VectorTools::L2_norm);
  const double u_l2_error_loc = cellwise_errors.norm_sqr();
  const double u_l2_error =
    std::sqrt(Utilities::MPI::sum(u_l2_error_loc, MPI_COMM_WORLD));

  // compute L2 norm of solution to get relative errors
  QGauss<dim> quadrature_2(v_degree);
  VectorTools::integrate_difference(navier_stokes.get_dof_handler_p(),
                                    navier_stokes.solution.block(1),
                                    Functions::ZeroFunction<dim>(1),
                                    cellwise_errors,
                                    quadrature_2,
                                    VectorTools::L2_norm);
  const double p_l2_norm =
    std::sqrt(Utilities::MPI::sum(cellwise_errors.norm_sqr(), MPI_COMM_WORLD));
  VectorTools::integrate_difference(navier_stokes.get_dof_handler_u(),
                                    navier_stokes.solution.block(0),
                                    Functions::ZeroFunction<dim>(dim),
                                    cellwise_errors,
                                    quadrature_2,
                                    VectorTools::L2_norm);
  const double u_l2_norm =
    std::sqrt(Utilities::MPI::sum(cellwise_errors.norm_sqr(), MPI_COMM_WORLD));

  std::cout.precision(4);
  pcout << "  L2-Errors absolute: ||e_p||_L2 = " << p_l2_error
        << ",   ||e_u||_L2 = " << u_l2_error << std::endl;
  pcout << "  L2-Errors relative: ||e_p||_L2 = " << p_l2_error / p_l2_norm
        << ",   ||e_u||_L2 = " << u_l2_error / u_l2_norm << std::endl;
  pcout << "  Cell divergence:    |div(u)|_cells = " << cell_div << std::endl;
  timer.leave_subsection();
}



template <int dim>
void
BeltramiProblem<dim>::output_results() const
{
  if (!navier_stokes.time_stepping.at_tick(
        navier_stokes.get_parameters().output_frequency))
    return;

  compute_errors();

  timer.enter_subsection("Output solution.");
  const double time = navier_stokes.time_stepping.now();
  exact.zero_out_ghost_values();
  VectorTools::interpolate(navier_stokes.get_dof_handler_u(),
                           ExactSolutionU<dim>(nu, time),
                           exact.block(0));
  navier_stokes.interpolate_pressure_field(ExactSolutionP<dim>(nu, time), exact.block(1));
  exact -= navier_stokes.solution;
  exact.update_ghost_values();
  navier_stokes.solution.update_ghost_values();

  /*
   *  we make sure that each processor only works
   *   on the subdomain it owns locally (and not on ghost or artificial cells)
   *   when building the joint solution vector. (cf step-32)
   *
   */
  DataOut<dim> data_out;

  std::vector<DataComponentInterpretation::DataComponentInterpretation>
    vector_component_interpretation(
      dim, DataComponentInterpretation::component_is_part_of_vector);
  data_out.add_data_vector(navier_stokes.get_dof_handler_u(),
                           navier_stokes.solution.block(0),
                           std::vector<std::string>(dim, "velocity"),
                           vector_component_interpretation);
  data_out.add_data_vector(navier_stokes.get_dof_handler_p(),
                           navier_stokes.solution.block(1),
                           "pressure");
  data_out.add_data_vector(navier_stokes.get_dof_handler_u(),
                           exact.block(0),
                           std::vector<std::string>(dim, "velocity_error"),
                           vector_component_interpretation);
  data_out.add_data_vector(navier_stokes.get_dof_handler_p(),
                           exact.block(1),
                           "pressure_error");
  data_out.build_patches();

  navier_stokes.write_data_output(navier_stokes.get_parameters().output_filename,
                                  navier_stokes.time_stepping,
                                  navier_stokes.get_parameters().output_frequency,
                                  triangulation,
                                  data_out);

  timer.leave_subsection();
}


template <int dim>
void
BeltramiProblem<dim>::run()
{
  timer.enter_subsection("Setup grid.");

  pcout << "Running a " << dim << "D Beltrami problem "
        << "using " << navier_stokes.time_stepping.name() << ", Q"
        << navier_stokes.get_fe_u().degree << "/Q" << navier_stokes.get_fe_p().degree
        << (navier_stokes.get_parameters().augmented_taylor_hood ? "+" : "")
        << " elements on " << Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)
        << " processes" << std::endl;

  const bool use_ball = false;
  {
    if (use_ball == false)
      {
        std::vector<unsigned int> subdivisions(dim, 4);
        subdivisions[0] = 4;

        const Point<dim> bottom_left =
          (dim == 2 ? Point<dim>(-1, -1) : Point<dim>(-1, -1, -1));
        const Point<dim> top_right = (dim == 2 ? Point<dim>(1, 1) : Point<dim>(1, 1, 1));

        GridGenerator::subdivided_hyper_rectangle(triangulation,
                                                  subdivisions,
                                                  bottom_left,
                                                  top_right);
      }
    else
      GridGenerator::hyper_ball(triangulation);
  }
  if (navier_stokes.get_parameters().global_refinements >= 2)
    triangulation.refine_global(navier_stokes.get_parameters().global_refinements - 2);

  typename Triangulation<dim>::active_cell_iterator cell = triangulation.begin_active();
  ++cell;
  ++cell;
  if (cell->is_locally_owned())
    cell->set_refine_flag();
  ++cell;
  if (cell->is_locally_owned())
    cell->set_refine_flag();
  triangulation.execute_coarsening_and_refinement();
  triangulation.refine_global(1);

  timer.leave_subsection();

  navier_stokes.set_velocity_dirichlet_boundary(
    0, std::make_shared<ExactSolutionU<dim>>(nu));

  navier_stokes.distribute_dofs();
  navier_stokes.print_n_dofs();

  const bool constrain_all_pressure_boundary = false;
  if (constrain_all_pressure_boundary)
    VectorTools::interpolate_boundary_values(navier_stokes.get_dof_handler_p(),
                                             0,
                                             Functions::ZeroFunction<dim>(1),
                                             navier_stokes.modify_constraints_p());
  else
    navier_stokes.fix_pressure_constant(0, std::make_shared<ExactSolutionP<dim>>(nu));

  navier_stokes.initialize_data_structures();
  navier_stokes.initialize_matrix_free();

  timer.enter_subsection("Setup initial condition.");

  VectorTools::interpolate(navier_stokes.get_dof_handler_u(),
                           ExactSolutionU<dim>(0.),
                           navier_stokes.solution.block(0));
  navier_stokes.interpolate_pressure_field(ExactSolutionP<dim>(nu),
                                           navier_stokes.solution.block(1));
  exact.reinit(navier_stokes.solution);

  navier_stokes.time_stepping.restart();
  timer.leave_subsection();

  // prepare exact solution for output
  output_results();

  // @sect5{Time loop}
  while (navier_stokes.time_stepping.at_end() == false)
    {
      navier_stokes.init_time_advance(true);

      const double time = navier_stokes.time_stepping.now();
      if (constrain_all_pressure_boundary)
        {
          std::map<types::global_dof_index, double> boundary_values;
          VectorTools::interpolate_boundary_values(navier_stokes.get_dof_handler_p(),
                                                   0,
                                                   ExactSolutionP<dim>(nu, time),
                                                   boundary_values);
          for (typename std::map<types::global_dof_index, double>::const_iterator it =
                 boundary_values.begin();
               it != boundary_values.end();
               ++it)
            if (navier_stokes.solution.block(1).in_local_range(it->first))
              navier_stokes.solution.block(1)(it->first) = it->second;
        }

      navier_stokes.evaluate_time_step();
      output_results();

      if (navier_stokes.time_stepping.step_no() % 10 == 0)
        {
          timer.print_summary();
          timer.reset();
        }
    }
  navier_stokes.print_memory_consumption();

  if (navier_stokes.time_stepping.at_tick(
        navier_stokes.get_parameters().output_frequency) == false)
    compute_errors();
}


/* ----------------------------------------------------------------------- */



/* ----------------------------------------------------------------------- */



int
main(int argc, char **argv)
{
  /* we initialize MPI at the start of the program. Since we will in general mix
   * MPI parallelization with threads, we also set the third argument in MPI_InitFinalize
   * that controls the number of threads to an invalid number, which means that the TBB
   * library chooses the number of threads automatically, typically to the number of
   * available cores in the system. As an alternative, you can also set this number
   * manually if you want to set a specific number of threads (e.g. when MPI-only is
   * required)  (cf step-40 and 48).
   */
  try
    {
      using namespace dealii;


      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, -1);
      deallog.depth_console(0);

      std::string paramfile;
      if (argc > 1)
        paramfile = argv[1];
      else
        paramfile = "beltrami.prm";

      FlowParameters parameters(paramfile);

      if (parameters.dimension == 2)
        {
          BeltramiProblem<2> beltrami_problem(parameters);
          beltrami_problem.run();
        }
      else if (parameters.dimension == 3)
        {
          BeltramiProblem<3> beltrami_problem(parameters);
          beltrami_problem.run();
        }
      else
        AssertThrow(false, ExcNotImplemented());
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------" << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------" << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------" << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------" << std::endl;
      return 1;
    }

  return 0;
}
