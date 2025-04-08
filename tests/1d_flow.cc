// --------------------------------------------------------------------------
//
// Copyright (C) 2021 by the adaflo authors
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

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/manifold_lib.h>
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
using namespace adaflo;

/**
 * Simple test case for 1d Navier-Stokes
 *
 *
 * velocity(t=0) = 2
 *                             L=2.5
 * (pressure=2) +-------------------------------+ (p=1)
 *              |--> x
 *
 */


template <int dim>
class ChannelFlow
{
public:
  ChannelFlow(const FlowParameters &parameters);
  void
  run();

private:
  void
  output_results() const;

  ConditionalOStream pcout;

  mutable TimerOutput timer;

  Triangulation<dim> triangulation;
  NavierStokes<dim>  navier_stokes;
};

template <int dim>
ChannelFlow<dim>::ChannelFlow(const FlowParameters &parameters)
  : pcout(std::cout, (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0))
  , timer(pcout, TimerOutput::summary, TimerOutput::cpu_and_wall_times)
  , navier_stokes(parameters, triangulation, &timer)
{}

template <int dim>
void
ChannelFlow<dim>::output_results() const
{
  timer.enter_subsection("Generate output.");

  navier_stokes.output_solution(navier_stokes.get_parameters().output_filename);

  timer.leave_subsection();
}


template <int dim>
void
create_triangulation(Triangulation<dim> &tria)
{
  AssertThrow(dim == 1, ExcNotImplemented());
  GridGenerator::hyper_rectangle(tria, Point<dim>(0.0), Point<dim>(2.5));
  tria.refine_global(10);

  for (const auto &cell : tria.cell_iterators())
    for (auto &face : cell->face_iterators())
      if (face->at_boundary())
        {
          if (std::abs(face->center()[0]) < 1e-12)
            face->set_boundary_id(0);
          else if (std::abs(face->center()[0] - 2.5) < 1e-12)
            face->set_boundary_id(1);
        }
}



template <int dim>
void
ChannelFlow<dim>::run()
{
  timer.enter_subsection("Setup grid and initial condition.");
  pcout << "Running a " << dim << "D flow "
        << "using " << navier_stokes.time_stepping.name() << ", Q"
        << navier_stokes.get_fe_u().degree << "/Q" << navier_stokes.get_fe_p().degree
        << " elements" << std::endl;

  create_triangulation(triangulation);

  navier_stokes.set_open_boundary_with_normal_flux(
    0, std::make_shared<Functions::ConstantFunction<dim>>(2));
  navier_stokes.set_open_boundary_with_normal_flux(
    1, std::make_shared<Functions::ConstantFunction<dim>>(1));

  timer.leave_subsection();

  navier_stokes.setup_problem(Functions::ConstantFunction<dim>(2));
  navier_stokes.print_n_dofs();

  output_results();

  while (navier_stokes.time_stepping.at_end() == false)
    {
      navier_stokes.advance_time_step();
      if (navier_stokes.time_stepping.at_tick(
            navier_stokes.get_parameters().output_frequency))
        output_results();
    }

  if (!navier_stokes.time_stepping.at_tick(
        navier_stokes.get_parameters().output_frequency))
    output_results();
}



int
main(int argc, char **argv)
{
  try
    {
      using namespace dealii;

      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc,
                                                          argv,
                                                          numbers::invalid_unsigned_int);
      deallog.depth_console(0);

      AssertDimension(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD), 1);

      std::string paramfile;
      if (argc > 1)
        paramfile = argv[1];
      else
        paramfile = "1d_flow.prm";

      FlowParameters parameters(paramfile);
      if (parameters.dimension == 1)
        {
          ChannelFlow<1> channel(parameters);
          channel.run();
        }
      else
        {
          AssertThrow(false, ExcMessage("Invalid dimension. Only 1D supported."));
        }
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
