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

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/function_time.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/parameter_handler.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

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



// @sect4{Exact Solution}

template <int dim>
class ExactSolutionU : public Function<dim>
{
public:
  ExactSolutionU (const double viscosity = 1.,
                  const double time = 0.)
    :
    Function<dim>(dim, time),
    nu(viscosity)
  {}

  virtual void vector_value (const Point<dim> &p,
                             Vector<double>   &values) const;

private:
  const double nu;
};

template <int dim>
void
ExactSolutionU<dim>::vector_value (const Point<dim> &p,
                                   Vector<double>   &values) const
{
  AssertDimension (values.size(), dim);

  // exact solution for channel flow in time-dependent setting is
  // approximately computed from an ODE of the form ds/dt + nu s - 1 = 0
  // with initial condition s(0) = 0, which gives the x-velocity of the form
  // u(x,y,t) = s(t) * (1-y)*(1+y) assuming a quadratic profile. But the
  // profile will only be quadratic in the steady state, so only use it for
  // the steady state
  // const double time = this->get_time();

  values(0) = 0.5/nu * (1-p[1])*(1+p[1]);
  for (unsigned int d=1; d<dim; ++d)
    values(d) = 0;
}




template <int dim>
class ExactSolutionP : public Function<dim>
{
public:
  ExactSolutionP ()
    :
    Function<dim>(1, 0)
  {}

  virtual double value (const Point<dim> &p,
                        const unsigned int) const;
};

template <int dim>
double
ExactSolutionP<dim>::value (const Point<dim> &p,
                            const unsigned int) const
{
  return 2-p[0];
}



// @sect3{The <code>Beltramiproblem</code> class template}
template <int dim>
class ChannelProblem
{

public:
  ChannelProblem (const FlowParameters &parameters);
  void run ();

private:
  void compute_errors () const;
  void output_results () const;

  ConditionalOStream  pcout;

  mutable TimerOutput timer;

  parallel::distributed::Triangulation<dim>   triangulation;
  NavierStokes<dim>    navier_stokes;
  const double         nu;

  const unsigned int output_timestep_skip;
};



template <int dim>
ChannelProblem<dim>::ChannelProblem (const FlowParameters &parameters)
  :
  pcout (std::cout,
         (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)),
  timer (pcout, TimerOutput::summary,
         TimerOutput::cpu_and_wall_times),
  triangulation(MPI_COMM_WORLD),
  navier_stokes (parameters, triangulation,
                 &timer),
  nu (parameters.viscosity),
  output_timestep_skip (4)
{}



template <int dim>
void ChannelProblem<dim>::compute_errors () const
{
  timer.enter_subsection("Compute errors.");

  Vector<float> cellwise_errors (triangulation.n_active_cells());
  const unsigned int v_degree =
    navier_stokes.get_parameters().velocity_degree;

  // use high order quadrature to avoid
  // underestimation of errors because of
  // superconvergence effects

  QGauss<dim>  quadrature(v_degree+2);

  // With this, we can then let the
  // library compute the errors and
  // output them to the screen:
  VectorTools::integrate_difference (navier_stokes.get_dof_handler_p(),
                                     navier_stokes.solution.block(1),
                                     ExactSolutionP<dim> (),
                                     cellwise_errors, quadrature,
                                     VectorTools::L2_norm);
  const double p_l2_error = cellwise_errors.l2_norm();

  VectorTools::integrate_difference (navier_stokes.get_dof_handler_u(),
                                     navier_stokes.solution.block(0),
                                     ExactSolutionU<dim> (nu, navier_stokes.time_stepping.now()),
                                     cellwise_errors, quadrature,
                                     VectorTools::L2_norm);
  const double u_l2_error = cellwise_errors.l2_norm();


  std::cout.precision(4);
  pcout << "  L2-Errors: ||e_p||_L2 = " << p_l2_error
        << ",   ||e_u||_L2 = " << u_l2_error
        << std::endl;

  timer.leave_subsection();
}



template <int dim>
void ChannelProblem<dim>::output_results () const
{
  ExactSolutionU<dim> exact(nu, navier_stokes.time_stepping.now());
  Vector<double> values(dim);
  exact.vector_value(Point<dim>(), values);

  pcout << "  Maximum velocity now: " << values[0] << std::endl;

  navier_stokes.output_solution(navier_stokes.get_parameters().output_filename);
}



template <int dim>
void ChannelProblem<dim>::run ()
{
  timer.enter_subsection ("Setup grid and initial condition.");
  pcout << "Running a " << dim << "D channel flow problem "
        << "using " << navier_stokes.time_stepping.name()
        << ", Q"  << navier_stokes.get_fe_u().degree
        << "/Q" << navier_stokes.get_fe_p().degree
        << " elements" << std::endl;

  {
    std::vector<unsigned int> subdivisions (dim, 1);
    subdivisions[0] = 4;

    const Point<dim> bottom_left = (dim == 2 ?
                                    Point<dim>(-2,-1) :
                                    Point<dim>(-2,-1,-1));
    const Point<dim> top_right   = (dim == 2 ?
                                    Point<dim>(2,0) :
                                    Point<dim>(2,0,0));

    GridGenerator::subdivided_hyper_rectangle (triangulation,
                                               subdivisions,
                                               bottom_left,
                                               top_right);

    // no need to check for owned cells here: on level 0 everything is locally
    // owned
    for (typename Triangulation<dim>::active_cell_iterator it=triangulation.begin();
         it != triangulation.end(); ++it)
      for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
        if (it->face(face)->at_boundary() &&
            std::abs(it->face(face)->center()[0]-2)<1e-13)
          it->face(face)->set_boundary_id(1);
        else if (it->face(face)->at_boundary() &&
                 std::abs(it->face(face)->center()[0]+2)<1e-13)
          it->face(face)->set_boundary_id(2);
        else if (it->face(face)->at_boundary() &&
                 std::abs(it->face(face)->center()[1])<1e-13)
          it->face(face)->set_boundary_id(3);
  }

  navier_stokes.set_no_slip_boundary(0);
  navier_stokes.set_symmetry_boundary(3);

  navier_stokes.set_open_boundary_with_normal_flux(1, std_cxx11::shared_ptr<Function<dim> > (new ExactSolutionP<dim>()));
  navier_stokes.set_open_boundary_with_normal_flux(2, std_cxx11::shared_ptr<Function<dim> > (new ExactSolutionP<dim>()));
  timer.leave_subsection();

  navier_stokes.setup_problem(ZeroFunction<dim>(dim));
  navier_stokes.print_n_dofs();
  output_results ();

  // @sect5{Time loop}
  if (navier_stokes.get_parameters().physical_type == FlowParameters::incompressible)
    while (navier_stokes.time_stepping.at_end() == false)
      {
        navier_stokes.advance_time_step();

        // We check whether we are at a time step where to save the current
        // solution to a file.
        if (navier_stokes.time_stepping.step_no() % output_timestep_skip == 0)
          {
            output_results ();
            compute_errors ();
          }
      }
  else
    navier_stokes.advance_time_step();

  if (navier_stokes.time_stepping.step_no() % output_timestep_skip != 0)
    compute_errors ();
}



/* ----------------------------------------------------------------------- */



/* ----------------------------------------------------------------------- */



int main (int argc, char **argv)
{
  /* we initialize MPI at the start of the program. Since we will in general mix
   * MPI parallelization with threads, we also set the third argument in MPI_InitFinalize
   * that controls the number of threads to an invalid number, which means that the TBB
   * library chooses the number of threads automatically, typically to the number of available
   * cores in the system. As an alternative, you can also set this number manually if you want
   * to set a specific number of threads (e.g. when MPI-only is required)  (cf step-40 and 48).
   */
  try
    {
      using namespace dealii;

      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);//numbers::invalid_unsigned_int);
      deallog.depth_console (0);

      std::string paramfile;
      if (argc>1)
        paramfile = argv[1];
      else
        paramfile = "channel.prm";

      FlowParameters parameters (paramfile);
      Assert(parameters.dimension == 2, ExcNotImplemented());

      ChannelProblem<2> channel(parameters);
      channel.run();
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
