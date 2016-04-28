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

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/function_time.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/parameter_handler.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/manifold_lib.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>


#include <deal.II/lac/parallel_vector.h>
#include <deal.II/lac/constraint_matrix.h>

#include <deal.II/base/index_set.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/grid_refinement.h>




#include <adaflo/navier_stokes.h>
#include <adaflo/time_stepping.h>

#include <fstream>
#include <iostream>
#include <iomanip>

using namespace dealii;



// @sect4{Inflow}

template <int dim>
class InflowVelocity : public Function<dim>
{
public:
  InflowVelocity (const double time,
                  const bool fluctuating)
    :
    Function<dim>(dim, time),
    fluctuating(fluctuating)
  {}

  virtual void vector_value (const Point<dim> &p,
                             Vector<double>   &values) const;

private:
  const bool fluctuating;
};

template <int dim>
void
InflowVelocity<dim>::vector_value (const Point<dim> &p,
                                   Vector<double>   &values) const
{
  AssertDimension (values.size(), dim);

  // inflow velocity according to Schaefer & Turek
  const double Um = (dim == 2 ? 1.5 : 2.25);
  const double H = 0.41;
  double coefficient = Utilities::fixed_power<dim-1>(4.) * Um / Utilities::fixed_power<2*dim-2>(H);
  values(0) = coefficient * p[1] * (H-p[1]);
  if (dim == 3)
    values(0) *= p[2] * (H-p[2]);
  for (unsigned int d=1; d<dim; ++d)
    values(d) = 0;
}



template <int dim>
class FlowPastCylinder
{

public:
  FlowPastCylinder (const FlowParameters &parameters);
  void run ();

private:
  void compute_statistics () const;
  void output_results () const;

  ConditionalOStream  pcout;

  mutable TimerOutput timer;

  std_cxx11::shared_ptr<Manifold<dim> > cylinder_manifold;
  parallel::distributed::Triangulation<dim>   triangulation;
  NavierStokes<dim>    navier_stokes;
};



namespace
{
  template <int dim>
  Point<dim> get_direction()
  {
    Point<dim> direction;
    direction[dim-1] = 1.;
    return direction;
  }

  template <int dim>
  Point<dim> get_center()
  {
    Point<dim> center;
    center[0] = 0.5;
    center[1] = 0.2;
    return center;
  }
}



template <int dim>
FlowPastCylinder<dim>::FlowPastCylinder (const FlowParameters &parameters)
  :
  pcout (std::cout,
         (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)),
  timer (pcout, TimerOutput::summary,
         TimerOutput::cpu_and_wall_times),
  cylinder_manifold(dim == 2 ?
                    static_cast<Manifold<dim>*>(new HyperBallBoundary<dim>(get_center<dim>(), 0.05)) :
                    static_cast<Manifold<dim>*>(new CylindricalManifold<dim>(get_direction<dim>(), get_center<dim>()))),
  triangulation(MPI_COMM_WORLD),
  navier_stokes (parameters, triangulation,
                 &timer)
{}



template <int dim>
void FlowPastCylinder<dim>::compute_statistics () const
{
  timer.enter_subsection("Compute statistics.");

  timer.leave_subsection();
}



template <int dim>
void FlowPastCylinder<dim>::output_results () const
{
  timer.enter_subsection("Generate output.");

  navier_stokes.output_solution(navier_stokes.get_parameters().output_filename);

  timer.leave_subsection();
}



void create_triangulation(Triangulation<2> &tria,
                          const bool compute_in_2d = true)
{
  HyperBallBoundary<2> boundary(Point<2>(0.5,0.2), 0.05);
  Triangulation<2> left, middle, right, tmp, tmp2;
  GridGenerator::subdivided_hyper_rectangle(left, std::vector<unsigned int>({3U, 4U}),
                                            Point<2>(), Point<2>(0.3, 0.41), false);
  GridGenerator::subdivided_hyper_rectangle(right, std::vector<unsigned int>({18U, 4U}),
                                            Point<2>(0.7, 0), Point<2>(2.5, 0.41), false);

  // create middle part first as a hyper shell
  GridGenerator::hyper_shell(middle, Point<2>(0.5, 0.2), 0.05, 0.2, 4, true);
  middle.set_manifold(0, boundary);
  middle.refine_global(1);

  // then move the vertices to the points where we want them to be to create a
  // slightly asymmetric cube with a hole
  for (Triangulation<2>::cell_iterator cell = middle.begin();
       cell != middle.end(); ++cell)
    for (unsigned int v=0; v < GeometryInfo<2>::vertices_per_cell; ++v)
      {
        Point<2> &vertex = cell->vertex(v);
        if (std::abs(vertex[0] - 0.7) < 1e-10 &&
            std::abs(vertex[1] - 0.2) < 1e-10)
          vertex = Point<2>(0.7, 0.205);
        else if (std::abs(vertex[0] - 0.6) < 1e-10 &&
                 std::abs(vertex[1] - 0.3) < 1e-10)
          vertex = Point<2>(0.7, 0.41);
        else if (std::abs(vertex[0] - 0.6) < 1e-10 &&
                 std::abs(vertex[1] - 0.1) < 1e-10)
          vertex = Point<2>(0.7, 0);
        else if (std::abs(vertex[0] - 0.5) < 1e-10 &&
                 std::abs(vertex[1] - 0.4) < 1e-10)
          vertex = Point<2>(0.5, 0.41);
        else if (std::abs(vertex[0] - 0.5) < 1e-10 &&
                 std::abs(vertex[1] - 0.0) < 1e-10)
          vertex = Point<2>(0.5, 0.0);
        else if (std::abs(vertex[0] - 0.4) < 1e-10 &&
                 std::abs(vertex[1] - 0.3) < 1e-10)
          vertex = Point<2>(0.3, 0.41);
        else if (std::abs(vertex[0] - 0.4) < 1e-10 &&
                 std::abs(vertex[1] - 0.1) < 1e-10)
          vertex = Point<2>(0.3, 0);
        else if (std::abs(vertex[0] - 0.3) < 1e-10 &&
                 std::abs(vertex[1] - 0.2) < 1e-10)
          vertex = Point<2>(0.3, 0.205);
        else if (std::abs(vertex[0] - 0.56379) < 1e-4 &&
                 std::abs(vertex[1] - 0.13621) < 1e-4)
          vertex = Point<2>(0.59, 0.11);
        else if (std::abs(vertex[0] - 0.56379) < 1e-4 &&
                 std::abs(vertex[1] - 0.26379) < 1e-4)
          vertex = Point<2>(0.59, 0.29);
        else if (std::abs(vertex[0] - 0.43621) < 1e-4 &&
                 std::abs(vertex[1] - 0.13621) < 1e-4)
          vertex = Point<2>(0.41, 0.11);
        else if (std::abs(vertex[0] - 0.43621) < 1e-4 &&
                 std::abs(vertex[1] - 0.26379) < 1e-4)
          vertex = Point<2>(0.41, 0.29);
      }

  // refine once to create the same level of refinement as in the
  // neighboring domains
  middle.refine_global(1);

  // must copy the triangulation because we cannot merge triangulations with
  // refinement...
  GridGenerator::flatten_triangulation(middle, tmp2);

  if (compute_in_2d)
    GridGenerator::merge_triangulations (tmp2, right, tria);
  else
    {
      GridGenerator::merge_triangulations (left, tmp2, tmp);
      GridGenerator::merge_triangulations (tmp, right, tria);
    }

  // Set the left boundary (inflow) to 1, the right to 2, the rest to 0.
  for (Triangulation<2>::active_cell_iterator cell=tria.begin() ;
       cell != tria.end(); ++cell)
    for (unsigned int f=0; f<GeometryInfo<2>::faces_per_cell; ++f)
      if (cell->face(f)->at_boundary())
        {
          if (std::abs(cell->face(f)->center()[0] - (compute_in_2d ? 0.3 : 0)) < 1e-12)
            cell->face(f)->set_all_boundary_ids(1);
          else if (std::abs(cell->face(f)->center()[0]-2.5) < 1e-12)
            cell->face(f)->set_all_boundary_ids(2);
          else if (Point<2>(0.5,0.2).distance(cell->face(f)->center())<=0.05)
            {
              cell->face(f)->set_all_manifold_ids(10);
              cell->face(f)->set_all_boundary_ids(0);
            }
          else
            cell->face(f)->set_all_boundary_ids(0);
        }
}



void create_triangulation(Triangulation<3> &tria)
{
  Triangulation<2> tria_2d;
  create_triangulation(tria_2d, false);
  GridGenerator::extrude_triangulation(tria_2d, 5, 0.41, tria);

  // set boundary indicators correctly
  for (Triangulation<3>::active_cell_iterator cell=tria.begin() ;
       cell != tria.end(); ++cell)
    for (unsigned int f=0; f<GeometryInfo<3>::faces_per_cell; ++f)
      if (cell->face(f)->at_boundary())
        {
          if (std::abs(cell->face(f)->center()[0]) < 1e-12)
            cell->face(f)->set_all_boundary_ids(1);
          else if (std::abs(cell->face(f)->center()[0]-2.5) < 1e-12)
            cell->face(f)->set_all_boundary_ids(2);
          else if (Point<3>(0.5,0.2,cell->face(f)->center()[2]).distance(cell->face(f)->center())<=0.05)
            {
              cell->face(f)->set_all_manifold_ids(10);
              cell->face(f)->set_all_boundary_ids(0);
            }
          else
            cell->face(f)->set_all_boundary_ids(0);
        }
}



template <int dim>
void FlowPastCylinder<dim>::run ()
{
  timer.enter_subsection ("Setup grid and initial condition.");
  pcout << "Running a " << dim << "D flow past a cylinder "
        << "using " << navier_stokes.time_stepping.name()
        << ", Q"  << navier_stokes.get_fe_u().degree
        << "/Q" << navier_stokes.get_fe_p().degree
        << " elements" << std::endl;

  create_triangulation(triangulation);
  triangulation.set_manifold(10, *cylinder_manifold);

  navier_stokes.set_no_slip_boundary(0);
  navier_stokes.set_velocity_dirichlet_boundary(1, std_cxx11::shared_ptr<Function<dim> >(new InflowVelocity<dim>(0., false)));

  navier_stokes.set_open_boundary(2, std_cxx11::shared_ptr<Function<dim> > (new ZeroFunction<dim>(1)));
  timer.leave_subsection();

  navier_stokes.setup_problem(InflowVelocity<dim>(0., false));
  navier_stokes.print_n_dofs();
  output_results ();

  // @sect5{Time loop}
  while (navier_stokes.time_stepping.at_end() == false)
    {
      navier_stokes.advance_time_step();

      // We check whether we are at a time step where to save the current
      // solution to a file.
      compute_statistics();
      if (navier_stokes.time_stepping.at_tick(navier_stokes.get_parameters().output_frequency))
        output_results ();
    }

  if (!navier_stokes.time_stepping.at_tick(navier_stokes.get_parameters().output_frequency))
    output_results ();
}



/* ----------------------------------------------------------------------- */



/* ----------------------------------------------------------------------- */



int main (int argc, char **argv)
{
  try
    {
      using namespace dealii;

      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv,
                                                          numbers::invalid_unsigned_int);
      deallog.depth_console (0);

      std::string paramfile;
      if (argc>1)
        paramfile = argv[1];
      else
        paramfile = "flow_past_cylinder.prm";

      FlowParameters parameters (paramfile);
      if (parameters.dimension == 2)
        {
          FlowPastCylinder<2> channel(parameters);
          channel.run();
        }
      else if (parameters.dimension == 3)
        {
          FlowPastCylinder<3> channel(parameters);
          channel.run();
        }
      else
        {
          AssertThrow(false, ExcMessage("Invalid dimension"));
        }
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}
