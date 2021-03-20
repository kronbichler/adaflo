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

template <int dim>
class CouetteProblem
{
public:
  CouetteProblem(const FlowParameters &parameters);
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
};



template <int dim>
CouetteProblem<dim>::CouetteProblem(const FlowParameters &parameters)
  : pcout(std::cout, (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0))
  , timer(pcout, TimerOutput::summary, TimerOutput::cpu_and_wall_times)
  , triangulation(MPI_COMM_WORLD)
  , navier_stokes(parameters, triangulation, &timer)
  , nu(parameters.viscosity)
{}



template <int dim>
void
CouetteProblem<dim>::output_results() const
{
  navier_stokes.output_solution(navier_stokes.get_parameters().output_filename);
}



template <int dim>
void
CouetteProblem<dim>::run()
{
  timer.enter_subsection("Setup grid and initial condition.");
  pcout << "Running a " << dim << "D Couette problem "
        << "using " << navier_stokes.time_stepping.name() << ", Q"
        << navier_stokes.get_fe_u().degree << "/Q" << navier_stokes.get_fe_p().degree
        << " elements" << std::endl;

  std::vector<unsigned int> subdivisions(dim, 1);
  subdivisions[0] = 4;

  const Point<dim> bottom_left = (dim == 2 ? Point<dim>(-2, -1) : Point<dim>(-2, -1, -1));
  const Point<dim> top_right   = (dim == 2 ? Point<dim>(2, 0) : Point<dim>(2, 0, 0));

  GridGenerator::subdivided_hyper_rectangle(triangulation,
                                            subdivisions,
                                            bottom_left,
                                            top_right);

  // no need to check for owned cells here: on level 0 everything is locally
  // owned
  for (typename Triangulation<dim>::active_cell_iterator it = triangulation.begin();
       it != triangulation.end();
       ++it)
    for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face)
      if (it->face(face)->at_boundary() &&
          std::abs(it->face(face)->center()[0] - 2) < 1e-13)
        it->face(face)->set_boundary_id(1); // left
      else if (it->face(face)->at_boundary() &&
               std::abs(it->face(face)->center()[0] + 2) < 1e-13)
        it->face(face)->set_boundary_id(2); // right
      else if (it->face(face)->at_boundary() &&
               std::abs(it->face(face)->center()[1]) < 1e-13)
        it->face(face)->set_boundary_id(3); // top


  navier_stokes.set_no_slip_boundary(0);

  const std::vector<double> vel = {2, 0};
  navier_stokes.set_velocity_dirichlet_boundary(
    3, std::make_shared<Functions::ConstantFunction<dim>>(vel));

  navier_stokes.set_open_boundary_with_normal_flux(
    1, std::make_shared<Functions::ZeroFunction<dim>>());
  navier_stokes.set_open_boundary_with_normal_flux(
    2, std::make_shared<Functions::ZeroFunction<dim>>());
  timer.leave_subsection();

  navier_stokes.setup_problem(Functions::ZeroFunction<dim>(dim));
  navier_stokes.print_n_dofs();
  output_results();

  if (navier_stokes.get_parameters().physical_type == FlowParameters::incompressible)
    while (navier_stokes.time_stepping.at_end() == false)
      {
        navier_stokes.advance_time_step();
        output_results();
      }
  else
    navier_stokes.advance_time_step();
}

/* ----------------------------------------------------------------------- */



/* ----------------------------------------------------------------------- */



int
main(int argc, char **argv)
{
  try
    {
      using namespace dealii;

      Utilities::MPI::MPI_InitFinalize mpi_initialization(
        argc, argv, 1); // numbers::invalid_unsigned_int);
      deallog.depth_console(0);

      std::string paramfile;
      if (argc > 1)
        paramfile = argv[1];
      else
        paramfile = "couette.prm";

      FlowParameters parameters(paramfile);
      Assert(parameters.dimension == 2, ExcNotImplemented());

      CouetteProblem<2> channel(parameters);
      channel.run();
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
