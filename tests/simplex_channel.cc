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

#include <deal.II/base/function.h>
#include <deal.II/base/function_time.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/utilities.h>

#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/shared_tria.h>

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

#include <deal.II/simplex/grid_generator.h>

#include <adaflo/navier_stokes.h>
#include <adaflo/time_stepping.h>

#include <fstream>
#include <iomanip>
#include <iostream>

#define MESH_VERSION 1

using namespace dealii;



// @sect4{Inflow}

template <int dim>
class InflowVelocity : public Function<dim>
{
public:
  InflowVelocity()
    : Function<dim>(dim)
  {}

  virtual void
  vector_value(const Point<dim> &p, Vector<double> &values) const
  {
    AssertDimension(values.size(), dim);

    (void)p;

    values(0) = 1;
    for (unsigned int d = 1; d < dim; ++d)
      values(d) = 0;
  }

private:
};


template <int dim>
class FlowPastCylinder
{
public:
  FlowPastCylinder(FlowParameters &parameters);
  void
  run();

private:
  void
  output_results() const;

  ConditionalOStream pcout;

  mutable TimerOutput timer;

  parallel::shared::Triangulation<dim> triangulation;

  const bool         use_simplex_mesh;
  const unsigned int n_refinements;

  NavierStokes<dim> navier_stokes;
};



namespace
{
  template <int dim>
  Point<dim>
  get_direction()
  {
    Point<dim> direction;
    direction[dim - 1] = 1.;
    return direction;
  }

  template <int dim>
  Point<dim>
  get_center()
  {
    Point<dim> center;
    center[0] = 0.5;
    center[1] = 0.2;
    return center;
  }

  unsigned int
  fix_n_refinements(FlowParameters &parameters)
  {
    unsigned int temp = parameters.global_refinements;

    parameters.global_refinements = 0;

    return temp;
  }

} // namespace



template <int dim>
FlowPastCylinder<dim>::FlowPastCylinder(FlowParameters &parameters)
  : pcout(std::cout, (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0))
  , timer(pcout, TimerOutput::summary, TimerOutput::cpu_and_wall_times)
  , triangulation(MPI_COMM_WORLD)
  , use_simplex_mesh(parameters.use_simplex_mesh)
  , n_refinements(fix_n_refinements(parameters))
  , navier_stokes(parameters, triangulation, &timer)
{}



template <int dim>
void
FlowPastCylinder<dim>::output_results() const
{
  timer.enter_subsection("Generate output.");

  navier_stokes.output_solution(navier_stokes.get_parameters().output_filename);

  timer.leave_subsection();
}



void create_triangulation(Triangulation<2> & tria,
                          const bool         use_simplex_mesh,
                          const unsigned int n_refinements)
{
  const unsigned int n = Utilities::pow(2, n_refinements);

  if (use_simplex_mesh)
    {
      GridGenerator::subdivided_hyper_rectangle_with_simplices(
        tria, {5 * n, 1 * n}, Point<2>(0.0, 0.0), Point<2>(5.0, 1.0), false);

      for (auto cell : tria)
        for (auto face : cell.face_iterators())
          {
            const auto p = face->center();

            if (std::abs(p[0] - 0.0) < 1e-8)
              face->set_boundary_id(0);
            else if (std::abs(p[0] - 5.0) < 1e-8)
              face->set_boundary_id(1);
            else if (std::abs(p[1] - 0.0) < 1e-8)
              face->set_boundary_id(2);
            else if (std::abs(p[1] - 1.0) < 1e-8)
              face->set_boundary_id(3);
          }
    }
  else
    {
      GridGenerator::subdivided_hyper_rectangle(
        tria, {5 * n, 1 * n}, Point<2>(0.0, 0.0), Point<2>(5.0, 1.0), true);
    }
}



void create_triangulation(Triangulation<3> & tria,
                          const bool         use_simplex_mesh,
                          const unsigned int n_refinements)
{
  const unsigned int n = Utilities::pow(2, n_refinements);

  if (use_simplex_mesh)
    {
      GridGenerator::subdivided_hyper_rectangle_with_simplices(tria,
                                                               {5 * n, 1 * n, 1 * n},
                                                               Point<3>(0.0, 0.0, 0.0),
                                                               Point<3>(5.0, 1.0, 1.0),
                                                               true);

      for (auto cell : tria)
        for (auto face : cell.face_iterators())
          {
            const auto p = face->center();

            if (std::abs(p[0] - 0.0) < 1e-8)
              face->set_boundary_id(0);
            else if (std::abs(p[0] - 5.0) < 1e-8)
              face->set_boundary_id(1);
            else if (std::abs(p[1] - 0.0) < 1e-8)
              face->set_boundary_id(2);
            else if (std::abs(p[1] - 1.0) < 1e-8)
              face->set_boundary_id(3);
            else if (std::abs(p[2] - 0.0) < 1e-8)
              face->set_boundary_id(4);
            else if (std::abs(p[2] - 1.0) < 1e-8)
              face->set_boundary_id(5);
          }
    }
  else
    {
      GridGenerator::subdivided_hyper_rectangle(tria,
                                                {5 * n, 1 * n, 1 * n},
                                                Point<3>(0.0, 0.0, 0.0),
                                                Point<3>(5.0, 1.0, 1.0),
                                                true);
    }
}



template <int dim>
void
FlowPastCylinder<dim>::run()
{
  timer.enter_subsection("Setup grid and initial condition.");
  pcout << "Running a " << dim << "D flow past a cylinder "
        << "using " << navier_stokes.time_stepping.name() << ", Q"
        << navier_stokes.get_fe_u().degree << "/Q" << navier_stokes.get_fe_p().degree
        << " elements" << std::endl;

  create_triangulation(triangulation, use_simplex_mesh, n_refinements);

  for (unsigned int i = 2; i < 2 * dim; ++i)
    navier_stokes.set_no_slip_boundary(i);

  navier_stokes.set_velocity_dirichlet_boundary(0,
                                                std::make_shared<InflowVelocity<dim>>());

  navier_stokes.set_open_boundary(1, std::make_shared<Functions::ZeroFunction<dim>>(1));
  timer.leave_subsection();

  navier_stokes.setup_problem(InflowVelocity<dim>());
  navier_stokes.print_n_dofs();
  output_results();

  // @sect5{Time loop}
  while (navier_stokes.time_stepping.at_end() == false)
    {
      navier_stokes.advance_time_step();

      // We check whether we are at a time step where to save the current
      // solution to a file.
      if (navier_stokes.time_stepping.at_tick(
            navier_stokes.get_parameters().output_frequency))
        output_results();
    }

  if (!navier_stokes.time_stepping.at_tick(
        navier_stokes.get_parameters().output_frequency))
    output_results();
}



/* ----------------------------------------------------------------------- */



/* ----------------------------------------------------------------------- */



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

      std::string paramfile;
      if (argc > 1)
        paramfile = argv[1];
      else
        paramfile = "flow_past_cylinder.prm";

      FlowParameters parameters(paramfile);
      if (parameters.dimension == 2)
        {
          FlowPastCylinder<2> channel(parameters);
          channel.run();
        }
      else if (parameters.dimension == 3)
        {
          Assert(false, ExcNotImplemented());
          /*
          FlowPastCylinder<3> channel(parameters);
          channel.run();
           */
        }
      else
        {
          AssertThrow(false, ExcMessage("Invalid dimension"));
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
