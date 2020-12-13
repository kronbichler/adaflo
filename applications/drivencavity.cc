// --------------------------------------------------------------------------
//
// Copyright (C) 2014 - 2016 by the adaflo authors
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
#include <deal.II/grid/grid_tools.h>
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


// @sect4{Boundary conditions: velocity ramped up over 1 time unit}


template <int dim>
class BoundaryVelocity : public Function<dim>
{
public:
  BoundaryVelocity(const double time = 0.)
    : Function<dim>(dim, time)
  {}

  virtual void
  vector_value(const Point<dim> &p, Vector<double> &values) const;
};

template <int dim>
void
BoundaryVelocity<dim>::vector_value(const Point<dim> &p, Vector<double> &values) const
{
  AssertDimension(values.size(), dim);

  // const double time = this->get_time ();

  for (unsigned int d = 0; d < dim; ++d)
    values(d) = 0;
  if (std::abs(p[1] - 1.) < 1e-12)
    {
      values(0) = 0.25 * (1 - std::cos(numbers::PI * 2 * p[0])) *
                  (1 - std::cos(numbers::PI * 2 * p[2] / 3.));
    }
}



// @sect3{The <code>LidDrivenCavityproblem</code> class template}
template <int dim>
class LidDrivenCavityProblem
{
public:
  LidDrivenCavityProblem(const FlowParameters &parameters);
  void
  run();

private:
  void
  output_results() const;

  void
  perform_data_exchange(std::vector<double> &        positions,
                        std::vector<Tensor<1, dim>> &velocities) const;

  ConditionalOStream pcout;

  mutable TimerOutput timer;

  parallel::distributed::Triangulation<dim> triangulation;
  NavierStokes<dim>                         navier_stokes;

  const double cavity_depth;
};



template <int dim>
LidDrivenCavityProblem<dim>::LidDrivenCavityProblem(const FlowParameters &parameters)
  : pcout(std::cout, (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0))
  , timer(pcout, TimerOutput::summary, TimerOutput::cpu_and_wall_times)
  , triangulation(MPI_COMM_WORLD)
  , navier_stokes(parameters, triangulation, &timer)
  , cavity_depth(3.)
{}



template <int dim>
void
LidDrivenCavityProblem<dim>::output_results() const
{
  timer.enter_subsection("Output solution.");
  navier_stokes.solution.update_ghost_values();

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
  data_out.build_patches(navier_stokes.get_parameters().velocity_degree);

  navier_stokes.write_data_output(navier_stokes.get_parameters().output_filename,
                                  navier_stokes.time_stepping,
                                  navier_stokes.get_parameters().output_frequency,
                                  this->triangulation,
                                  data_out);

  // Also write out the velocity along two lines at the middle of the cavity
  // We store data in a chunk of in total 7 rows: first the y/z coordinates of
  // the points, then the xyz velocities along an x stripe at y=1/2,
  // z=length/2 and then the xyz velocities along a y stripe at x=1/2, z=length/2
  const double x_pos = 0.5, zpos = cavity_depth * 0.5, y_pos = 0.5;

  std::vector<Point<dim - 1>> evaluate_points(
    navier_stokes.get_parameters().velocity_degree * 3 + 4);
  for (unsigned int i = 0; i < evaluate_points.size(); ++i)
    evaluate_points[i][0] = (double)i / (evaluate_points.size() - 1);
  Quadrature<dim - 1> quadrature(evaluate_points);
  FEFaceValues<dim>   fe_face_values(navier_stokes.get_dof_handler_u().get_fe(),
                                   quadrature,
                                   update_values | update_quadrature_points);

  std::vector<Tensor<1, dim>> velocities(evaluate_points.size());
  std::vector<Tensor<1, dim>> velocities_along_x;
  std::vector<Tensor<1, dim>> velocities_along_y;
  std::vector<double>         positions_along_x;
  std::vector<double>         positions_along_y;

  FEValuesExtractors::Vector velocity(0);
  typename DoFHandler<dim>::active_cell_iterator
    cell = navier_stokes.get_dof_handler_u().begin_active(),
    endc = navier_stokes.get_dof_handler_u().end();
  for (; cell != endc; ++cell)
    if (cell->is_locally_owned())
      {
        if (std::abs(cell->vertex(0)[0] - x_pos) < 1e-12 &&
            std::abs(cell->vertex(0)[2] - zpos) < 1e-12)
          {
            fe_face_values.reinit(cell, 0);
            fe_face_values[velocity].get_function_values(navier_stokes.solution.block(0),
                                                         velocities);
            // assume that the order of traversal in the center and at the
            // boundary is the same, which is ok for simple Cartesian meshes
            // (even those generated by subdivided_hypercube/hyperrectanlge I
            // think)
            velocities_along_y.insert(velocities_along_y.end(),
                                      velocities.begin(),
                                      velocities.end());
            for (unsigned int q = 0; q < fe_face_values.n_quadrature_points; ++q)
              positions_along_y.push_back(fe_face_values.quadrature_point(q)[1]);
          }
        if (std::abs(cell->vertex(0)[1] - y_pos) < 1e-12 &&
            std::abs(cell->vertex(0)[2] - zpos) < 1e-12)
          {
            fe_face_values.reinit(cell, 4);
            fe_face_values[velocity].get_function_values(navier_stokes.solution.block(0),
                                                         velocities);
            velocities_along_x.insert(velocities_along_x.end(),
                                      velocities.begin(),
                                      velocities.end());
            for (unsigned int q = 0; q < fe_face_values.n_quadrature_points; ++q)
              positions_along_x.push_back(fe_face_values.quadrature_point(q)[0]);
          }
      }

  perform_data_exchange(positions_along_x, velocities_along_x);
  perform_data_exchange(positions_along_y, velocities_along_y);

  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) > 0)
    return;

  std::string filename =
    navier_stokes.get_parameters().output_filename + "-" +
    Utilities::int_to_string(navier_stokes.get_parameters().global_refinements);
  if (std::abs(triangulation.begin_active()->diameter() -
               (++triangulation.begin_active())->diameter()) > 1e-12)
    filename += "-STR";
  if (triangulation.n_levels() > 3)
    filename += "-A" + Utilities::int_to_string(triangulation.n_levels() - 3);
  filename += "-Q" +
              Utilities::int_to_string(navier_stokes.get_parameters().velocity_degree) +
              "-statistics.dat";
  std::cout << std::endl
            << std::endl
            << "filename: " << filename << std::endl
            << std::endl
            << std::endl;
  std::ofstream output(filename.c_str());
  output.precision(8);
  for (unsigned int i = 0; i < positions_along_x.size(); ++i)
    output << positions_along_x[i] << " ";
  output << std::endl;
  for (unsigned int d = 0; d < dim; ++d)
    {
      for (unsigned int i = 0; i < velocities_along_x.size(); ++i)
        output << velocities_along_x[i][d] << " ";
      output << std::endl;
    }
  for (unsigned int i = 0; i < positions_along_y.size(); ++i)
    output << positions_along_y[i] << " ";
  output << std::endl;
  for (unsigned int d = 0; d < dim; ++d)
    {
      for (unsigned int i = 0; i < velocities_along_y.size(); ++i)
        output << velocities_along_y[i][d] << " ";
      output << std::endl;
    }

  timer.leave_subsection();
}



template <int dim>
void
LidDrivenCavityProblem<dim>::perform_data_exchange(
  std::vector<double> &        positions,
  std::vector<Tensor<1, dim>> &velocities) const
{
  // Exchange data with other processors
  std::vector<unsigned int> receive_count(
    Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD));

  unsigned int n_send_elements = velocities.size();
  MPI_Gather(&n_send_elements,
             1,
             MPI_UNSIGNED,
             &receive_count[0],
             1,
             MPI_UNSIGNED,
             0,
             MPI_COMM_WORLD);
  for (unsigned int i = 1; i < Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD); ++i)
    {
      // Each processor sends the interface_points he deals with to processor
      // 0
      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == i)
        {
          // put data into a std::vector<double> to create a data type that
          // MPI understands
          std::vector<double> send_data((dim + 1) * velocities.size());
          for (unsigned int j = 0; j < velocities.size(); ++j)
            {
              for (unsigned int d = 0; d < dim; ++d)
                send_data[j * (dim + 1) + d] = velocities[j][d];
              send_data[j * (dim + 1) + dim] = positions[j];
            }

          MPI_Send(&send_data[0], send_data.size(), MPI_DOUBLE, 0, i, MPI_COMM_WORLD);
        }

      // Processor 0 receives data from the other processors
      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        {
          std::vector<double> receive_data((dim + 1) * receive_count[i]);
          int                 ierr = MPI_Recv(&receive_data[0],
                              receive_data.size(),
                              MPI_DOUBLE,
                              i,
                              i,
                              MPI_COMM_WORLD,
                              MPI_STATUSES_IGNORE);
          AssertThrow(ierr == MPI_SUCCESS, ExcInternalError());
          for (unsigned int j = 0; j < receive_count[i]; ++j)
            {
              Tensor<1, dim> velocity;
              for (unsigned int d = 0; d < dim; ++d)
                velocity[d] = receive_data[j * (dim + 1) + d];
              positions.push_back(receive_data[j * (dim + 1) + dim]);
              velocities.push_back(velocity);
            }
        }
    }
}



template <int dim>
Point<dim>
grid_transform(const Point<dim> &in)
{
  Assert(dim == 2 || dim == 3, ExcNotImplemented());
  Point<dim> out = in;
  out[0]         = 0.5 + 0.5 * std::tanh(2 * (2. * in(0) - 1)) / std::tanh(2);
  out[1]         = 0.5 + 0.5 * std::tanh(2 * (2. * in(1) - 1)) / std::tanh(2);
  return out;
}



template <int dim>
void
LidDrivenCavityProblem<dim>::run()
{
  timer.enter_subsection("Setup grid and initial condition.");

  pcout << "Running a " << dim << "D lid driven cavity problem "
        << "using " << navier_stokes.time_stepping.name() << ", Q"
        << navier_stokes.get_fe_u().degree << "/Q" << navier_stokes.get_fe_p().degree
        << (navier_stokes.get_parameters().augmented_taylor_hood ? "+" : "")
        << " elements on " << Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)
        << " processes" << std::endl;

  {
    Point<dim> point;
    point[0] = point[1] = 1.;
    if (dim == 3)
      point[2] = cavity_depth;
    AssertThrow(navier_stokes.get_parameters().global_refinements % 4 == 0,
                ExcMessage(
                  "The number of elements per direction must be divisible by 4"));
    std::vector<unsigned int> refinements(
      dim, navier_stokes.get_parameters().global_refinements / 4);
    GridGenerator::subdivided_hyper_rectangle(triangulation,
                                              refinements,
                                              Point<dim>(),
                                              point);
    triangulation.refine_global(2);

    // GridTools::transform (&grid_transform<dim>, triangulation);
  }

  navier_stokes.set_velocity_dirichlet_boundary(
    0, std::make_shared<BoundaryVelocity<dim>>());
  navier_stokes.fix_pressure_constant(0,
                                      std::make_shared<Functions::ZeroFunction<dim>>());

  navier_stokes.distribute_dofs();

  timer.leave_subsection();

  unsigned int adaptive_refinements =
    navier_stokes.get_parameters().adaptive_refinements + 1;

  navier_stokes.time_stepping.restart();
  while (adaptive_refinements > 0)
    {
      navier_stokes.print_n_dofs();

      navier_stokes.initialize_data_structures();
      navier_stokes.initialize_matrix_free();

      // @sect5{Nonlinear solver}
      navier_stokes.advance_time_step();
      // while (navier_stokes.time_stepping.at_end() == false)
      //   {
      //     navier_stokes.init_time_advance (true);

      //     const double initial_residual = navier_stokes.compute_initial_residual(true);
      //     navier_stokes.solve_nonlinear_system(initial_residual);

      //     if (initial_residual < 10. * navier_stokes.get_parameters().tol_nl_iteration)
      //       break;
      //   }

      output_results();
      navier_stokes.print_memory_consumption();

      navier_stokes.refine_grid_pressure_based(100, 0.1, 0);
      --adaptive_refinements;
    }
}


/* ----------------------------------------------------------------------- */



/* ----------------------------------------------------------------------- */



int
main(int argc, char **argv)
{
  /* we initialize MPI at the start of the program. Since we will in general
   * mix MPI parallelization with threads, we also set the third argument in
   * MPI_InitFinalize that controls the number of threads to an invalid
   * number, which means that the TBB library chooses the number of threads
   * automatically, typically to the number of available cores in the
   * system. As an alternative, you can also set this number manually if you
   * want to set a specific number of threads (e.g. when MPI-only is required)
   * (cf step-40 and 48).
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
        paramfile = "drivencavity.prm";

      FlowParameters parameters(paramfile);

      if (parameters.dimension == 2)
        {
          LidDrivenCavityProblem<2> beltrami_problem(parameters);
          beltrami_problem.run();
        }
      else if (parameters.dimension == 3)
        {
          LidDrivenCavityProblem<3> beltrami_problem(parameters);
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
